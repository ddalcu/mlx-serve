#!/bin/bash
# test_hybrid_reuse_equivalence.sh — Phase 1 of performance-plan.md.
#
# Hybrid SSM models (qwen3_5, qwen3_5_moe, qwen3_next, lfm2, nemotron_h)
# went from "ZERO multi-turn prefix reuse" to "stride-aligned warm reuse via
# per-position SSM/conv state checkpoints". This test pins three invariants
# against future regressions:
#
#  1. Warm output is byte-identical to cold (temperature=0, same prompt).
#  2. `cached_n` on the second identical request is > 0 (proof the cache
#     actually engaged — the bug we're guarding against silently disabled
#     itself if anything goes wrong in lookupAndRestore).
#  3. Warm `prompt_ms` is at most 0.7× cold `prompt_ms` (Phase 1 acceptance:
#     "turn 2..5 prefill compute < 20% of turn-1's" for the all-shared case;
#     0.7× is a loose, hardware-agnostic threshold).
#
# Env:
#   MLX_HYBRID_MODEL  Path to a hybrid MLX model dir. Default: Qwen3.5-4B-MLX-4bit
#                     under Sandisk; pass another to exercise lfm2/qwen3_next/etc.
#   PORT              Server port. Default 19077.
#
# Reverting Phase 1 (in src/prefix_cache.zig or src/generate.zig) must
# turn this test red.

set -uo pipefail

MODEL="${MLX_HYBRID_MODEL:-/Volumes/Sandisk_1TB/Models/mlx-community/Qwen3.5-4B-MLX-4bit}"
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
trap 'kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; true' EXIT

"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 \
    --prefix-cache-entries 4 --ssm-checkpoint-stride 128 \
    --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
    sleep 0.5
done

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
        -d "$(jq -nc --arg p "$PROMPT" '{messages:[{role:"user",content:$p}],max_tokens:40,temperature:0,stream:false}')"
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
