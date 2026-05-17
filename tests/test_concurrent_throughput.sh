#!/bin/bash
# Concurrent decode throughput test (Phase A7).
#
# Asserts that with `--max-concurrent 4`, four parallel client requests
# generate aggregate tokens-per-second meaningfully faster than a single
# request running alone — the whole point of continuous batching. Threshold:
# aggregate_tps >= 1.6 * single_stream_tps. (At full pipelining we'd expect
# closer to 4×, but real-world overhead — prefill serialization, sampling per
# slot, attention scaling with N — pushes the achievable speedup down. 1.6×
# catches "batching is broken / completely serialized" without being so tight
# that variance flips the test.)
#
# Single-stream baseline runs first against the same server config, on the
# same prompt, with the same max_tokens, to keep the numerator and
# denominator comparable.
#
# Requires:
#   - A built mlx-serve binary
#   - Either CONCURRENT_TEST_MODEL set or default at
#     ~/.mlx-serve/models/gemma-4-e4b-it-8bit
#
# Usage:
#   CONCURRENT_TEST_MODEL=/path/to/model ./tests/test_concurrent_throughput.sh [port]

set -e

PORT=${1:-8093}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${CONCURRENT_TEST_MODEL:-${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}}"
MAX_TOKENS=${MAX_TOKENS:-200}
SPEEDUP_THRESHOLD=${SPEEDUP_THRESHOLD:-1.6}
N_PARALLEL=${N_PARALLEL:-4}

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_concurrent_throughput: model directory not found."
    echo "  Set CONCURRENT_TEST_MODEL or place an MLX checkpoint at"
    echo "  ~/.mlx-serve/models/gemma-4-e4b-it-8bit."
    exit 0
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable."
    exit 1
fi

# Skip MoE / hybrid / encoder-only models — those clamp max_concurrent to 1
# at server start because the batched kernel doesn't model their state. The
# test wouldn't be measuring continuous batching on them.
ARCH=$(python3 -c "
import json, sys
with open('$MODEL/config.json') as f:
    c = json.load(f)
mt = c.get('model_type', '')
moe_layers = c.get('num_local_experts', 0) > 0 or c.get('num_experts', 0) > 0
hybrid = mt in ('qwen3_5', 'qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_next', 'nemotron_h', 'lfm2', 'lfm2_vl')
encoder = c.get('is_encoder_only', False) or 'bert' in mt.lower()
if moe_layers or hybrid or encoder:
    print(f'SKIP_INCOMPATIBLE {mt}')
else:
    print(f'OK {mt}')
")
if [[ "$ARCH" == SKIP_INCOMPATIBLE* ]]; then
    arch_name="${ARCH#SKIP_INCOMPATIBLE }"
    echo -e "${YELLOW}SKIP${NC} test_concurrent_throughput: model arch '$arch_name' clamps max_concurrent to 1."
    echo "  Continuous batching only applies to pure-attention models. Use a"
    echo "  Gemma-4 / Llama / Mistral / Qwen3 dense checkpoint to exercise it."
    exit 0
fi

# Decode-bound prompt — short input, long output. We want the timing to be
# dominated by per-token decode work so the batching benefit is visible.
PROMPT='Write a detailed essay about quantum computing'

JSON_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': $MAX_TOKENS,
    'temperature': 0.0,
    'stream': False,
}))
")

echo "== concurrent throughput test =="
echo "  model: $MODEL"
echo "  max_tokens: $MAX_TOKENS, parallel: $N_PARALLEL, threshold: ${SPEEDUP_THRESHOLD}× speedup"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --max-concurrent "$N_PARALLEL" --no-pld > "$LOGFILE" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; rm -f $LOGFILE" EXIT INT TERM

echo "  starting server (--max-concurrent $N_PARALLEL)..."
up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -30 "$LOGFILE"
    exit 1
fi

# Confirm the server actually accepted the concurrent setting (didn't clamp
# to 1 because of an unexpected arch detection).
if grep -q "Concurrency: requested .* falling back to 1" "$LOGFILE"; then
    echo -e "${YELLOW}SKIP${NC} server clamped max_concurrent to 1:"
    grep "Concurrency:" "$LOGFILE" | sed 's/^/    /'
    exit 0
fi

# ── Single-stream baseline ──
# Warm-up first (drops cold-prefill, JIT, etc. out of the measurement).
echo "  warmup..."
echo "$JSON_PAYLOAD" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" > /dev/null

echo "  measuring single-stream baseline..."
T0=$(python3 -c 'import time; print(time.time())')
SINGLE_RESP=$(echo "$JSON_PAYLOAD" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
T1=$(python3 -c 'import time; print(time.time())')
SINGLE_TOKS=$(echo "$SINGLE_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
SINGLE_ELAPSED=$(python3 -c "print($T1 - $T0)")
SINGLE_TPS=$(python3 -c "print($SINGLE_TOKS / max($SINGLE_ELAPSED, 0.001))")
echo "  single: ${SINGLE_TOKS} toks in ${SINGLE_ELAPSED}s → ${SINGLE_TPS} tok/s"

# ── Parallel measurement ──
# Fire $N_PARALLEL requests in parallel. Capture per-request elapsed + token
# counts. Aggregate tps = sum(tokens) / wall_clock_elapsed.
echo "  measuring ${N_PARALLEL}-way concurrent..."

TMPDIR=$(mktemp -d)
T0=$(python3 -c 'import time; print(time.time())')
# Capture subshell PIDs and wait for THOSE, not bare `wait` which would also
# wait on the backgrounded server we started earlier.
PIDS=()
for i in $(seq 1 "$N_PARALLEL"); do
    (
        # Salt each prompt slightly so they don't all hit the exact same hot
        # cache entry. Use the request index so logs distinguish them.
        SALTED=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '[req $i] $PROMPT'}],
    'max_tokens': $MAX_TOKENS,
    'temperature': 0.0,
    'stream': False,
}))
")
        RESP=$(echo "$SALTED" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
        echo "$RESP" | python3 -c "import sys, json; r=json.load(sys.stdin); print(r['usage']['completion_tokens'])" > "$TMPDIR/req_$i.toks"
    ) &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
T1=$(python3 -c 'import time; print(time.time())')
WALL_ELAPSED=$(python3 -c "print($T1 - $T0)")

TOTAL_TOKS=0
for f in "$TMPDIR"/req_*.toks; do
    n=$(cat "$f")
    TOTAL_TOKS=$((TOTAL_TOKS + n))
done
rm -rf "$TMPDIR"

AGG_TPS=$(python3 -c "print($TOTAL_TOKS / max($WALL_ELAPSED, 0.001))")
SPEEDUP=$(python3 -c "print(round($AGG_TPS / max($SINGLE_TPS, 0.001), 2))")

echo "  parallel: ${TOTAL_TOKS} toks total in ${WALL_ELAPSED}s → ${AGG_TPS} tok/s aggregate"
echo "  speedup: ${SPEEDUP}× (threshold ${SPEEDUP_THRESHOLD}×)"

PASS=$(python3 -c "print('1' if $SPEEDUP >= $SPEEDUP_THRESHOLD else '0')")
if [ "$PASS" = "1" ]; then
    echo -e "${GREEN}PASS${NC} concurrent throughput speedup ${SPEEDUP}× >= ${SPEEDUP_THRESHOLD}×"
    exit 0
else
    echo -e "${RED}FAIL${NC} concurrent throughput speedup ${SPEEDUP}× < ${SPEEDUP_THRESHOLD}×"
    echo "  diagnostic: tail of server log:"
    tail -20 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi
