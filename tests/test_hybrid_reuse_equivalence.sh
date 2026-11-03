#!/bin/bash
# test_hybrid_reuse_equivalence.sh — prefix reuse on hybrid SSM models (qwen3_5, qwen3_5_moe, qwen3_next, lfm2,
# nemotron_h) through per-position SSM/conv checkpoints: [1] warm output is byte-identical to cold (temp 0, and seeded
# temp 1, whose draws key on absolute positions); [2] cached_n > 0 on the second identical request (the cache
# engaged); [3] warm prompt_ms <= 0.7x cold.
#
#   MLX_HYBRID_MODEL=<dir> ./tests/test_hybrid_reuse_equivalence.sh   # default Qwen3.5-4B-MLX-4bit; PORT (19077)

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

source "$(dirname "$0")/_lib_models.sh"
MODEL="${MLX_HYBRID_MODEL:-$(find_model mlx-community/Qwen3.5-4B-MLX-4bit)}"
PORT="${PORT:-19077}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

# Ensure no other server is running on the port.
pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
sleep 1

LOG="$(mktemp)"
SERVER_PID=""   # seeded: the EXIT trap runs on the SKIP path too, before any server starts
trap 'kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; true' EXIT

"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 \
    --prefix-cache-entries 4 --ssm-checkpoint-stride 128 \
    --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!

srv_wait_health "$BASE" 120 0.5 "$SERVER_PID" || kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }

# Build a long-ish prompt so the cache actually hits stride boundaries.
PROMPT="$(python3 - <<'PY'
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Continue the story in exactly 4 sentences.\n\nBackground:\n" + para * 50)
PY
)"

req_one() {
    curl -sf --max-time 60 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PROMPT" --argjson t "${1:-0}" '{messages:[{role:"user",content:$p}],max_tokens:40,temperature:$t,top_k:20,seed:7,stream:false}')"
}

echo "--- cold ---"
COLD_RESP=$(req_one)
[ -n "$COLD_RESP" ] || { echo "fail: cold request returned nothing"; tail -20 "$LOG"; exit 1; }
COLD_TEXT=$(echo "$COLD_RESP" | jq -r '.choices[0].message.content')
COLD_MS=$(echo "$COLD_RESP" | jq -r '.timings.prompt_ms')
COLD_CACHED=$(echo "$COLD_RESP" | jq -r '.timings.cached_n')
echo "  prompt_ms=$COLD_MS cached_n=$COLD_CACHED"
echo "  text=$(echo "$COLD_TEXT" | head -c 100)..."

echo "--- warm (identical prompt) ---"
WARM_RESP=$(req_one)
WARM_TEXT=$(echo "$WARM_RESP" | jq -r '.choices[0].message.content')
WARM_MS=$(echo "$WARM_RESP" | jq -r '.timings.prompt_ms')
WARM_CACHED=$(echo "$WARM_RESP" | jq -r '.timings.cached_n')
echo "  prompt_ms=$WARM_MS cached_n=$WARM_CACHED"
echo "  text=$(echo "$WARM_TEXT" | head -c 100)..."

EC=0
if [ "$COLD_TEXT" != "$WARM_TEXT" ]; then
    echo "❌ FAIL: warm output diverged from cold"
    diff <(printf "%s" "$COLD_TEXT") <(printf "%s" "$WARM_TEXT") | head -20
    EC=1
else
    echo "✅ byte-identical output"
fi

echo "--- seeded temperature=1, cold then warm ---"
PROMPT="Seeded. $PROMPT"
SEED_COLD=$(req_one 1 | jq -r '.choices[0].message.content')
SEED_WARM=$(req_one 1 | jq -r '.choices[0].message.content')
if [ -z "$SEED_COLD" ] || [ "$SEED_COLD" != "$SEED_WARM" ]; then
    echo "❌ FAIL: seeded warm output diverged from cold"
    diff <(printf "%s" "$SEED_COLD") <(printf "%s" "$SEED_WARM") | head -20
    EC=1
else
    echo "✅ seeded byte-identical output"
fi

if ! python3 -c "import sys; sys.exit(0 if $WARM_CACHED > 0 else 1)"; then
    echo "❌ FAIL: warm cached_n=$WARM_CACHED (expected > 0). SSM checkpoint cache didn't engage."
    EC=1
else
    echo "✅ warm reuse engaged (cached_n=$WARM_CACHED)"
fi

# 0.7× is loose to avoid flakiness on different hardware. The real win is
# much bigger (~5× on M-series at 1k+ tokens), but we want a stable gate.
if ! python3 -c "import sys; sys.exit(0 if float('$WARM_MS') < 0.7 * float('$COLD_MS') else 1)"; then
    echo "❌ FAIL: warm prompt_ms=$WARM_MS not <70% of cold prompt_ms=$COLD_MS"
    EC=1
else
    SPEEDUP=$(python3 -c "print(f'{float(\"$COLD_MS\") / max(float(\"$WARM_MS\"), 1e-6):.2f}x')")
    echo "✅ warm TTFT $SPEEDUP faster (warm=${WARM_MS}ms vs cold=${COLD_MS}ms)"
fi

exit $EC
