#!/bin/bash
# bench_prefill.sh — measure cold vs warm prefill throughput (TTFT root cause).
#
# Prefill (prompt processing) is what makes long-prompt / multi-turn requests
# feel slow: client-side TTFT conflates tokenize + Jinja + prefill + first
# decode. This harness isolates *prefill compute* using the server's own
# per-request timing (Phase 1: `timings.prompt_per_second` over the UNCACHED
# tokens, plus `timings.cached_n` for prefix reuse) so we can tell a cold
# prefill from a warm cache hit.
#
# Two scenarios, same model/server:
#   COLD — every request gets a unique nonce at the very front, so the prompt
#          shares no prefix with the previous one (cached_n ≈ 0). Measures raw
#          prefill tok/s.
#   WARM — the same long prompt is sent twice; the second reuses the resident
#          KV prefix (llama persistent session, or MLX hot prefix cache for
#          plain-attention models). Measures the cache hit (cached_n high,
#          tok/s reflects only the tiny suffix).
#
# Works for any model the binary auto-routes (GGUF file OR MLX dir).
#
# Usage:
#   ./tests/bench_prefill.sh <model_path> [port] [runs] [prompt_tokens]
# Examples:
#   ./tests/bench_prefill.sh ~/Models/Qwen3.5-4B-GGUF/Qwen3.5-4B-IQ4_NL.gguf
#   ./tests/bench_prefill.sh ~/Models/mlx-community/Qwen3.5-4B-MLX-4bit 19044
set -uo pipefail

MODEL="${1:-}"
PORT="${2:-19044}"
RUNS="${3:-3}"
PROMPT_TOKENS="${4:-750}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -n "$MODEL" ] || { echo "usage: $0 <model_path> [port] [runs] [prompt_tokens]" >&2; exit 1; }
[ -e "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -x "$BIN" ]   || { echo "build first: zig build -Doptimize=ReleaseFast" >&2; exit 1; }
command -v jq >/dev/null || { echo "needs jq" >&2; exit 1; }

trap 'kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; true' EXIT
LOG="$(mktemp)"

echo "=== prefill bench: $(basename "$MODEL") (runs=$RUNS, ~${PROMPT_TOKENS} prompt tokens) ==="
# --prefix-cache-entries keeps the MLX hot cache on for the warm scenario; the
# llama path uses its persistent session regardless.
"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 \
    --prefix-cache-entries 8 --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 240); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:"; tail -20 "$LOG"; exit 1; }
    sleep 0.5
done

# A ~PROMPT_TOKENS-token base paragraph (each repeat ≈ 15 tokens), plus a
# request builder that optionally prepends a unique nonce at the front.
post_prompt() {  # $1 = full user content; echoes the timings JSON object
    local body
    body=$(jq -nc --arg c "$1" \
        '{messages:[{role:"user",content:$c}],max_tokens:4,temperature:0,stream:false}')
    curl -sf --max-time 120 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" | jq -c '.timings // empty'
}

base_prompt() {
    python3 - "$PROMPT_TOKENS" <<'PY'
import sys
n = int(sys.argv[1])
sentence = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
# ~13 tokens/sentence; repeat to roughly hit the target token budget.
reps = max(1, n // 13)
sys.stdout.write("Summarize the following text in one word.\n\n" + sentence * reps)
PY
}

med() { python3 -c "import sys;xs=sorted(float(x) for x in sys.argv[1].split(',') if x);print(f'{xs[len(xs)//2]:.1f}' if xs else '0')" "$1"; }

BASE_PROMPT="$(base_prompt)"

echo
echo "--- COLD (unique prompt each run, no prefix reuse) ---"
cold_tps=""; cold_cached=""
for r in $(seq 1 "$RUNS"); do
    nonce="run-$r-$RANDOM-$RANDOM unique preamble. "
    t="$(post_prompt "${nonce}${BASE_PROMPT}")"
    [ -z "$t" ] && { echo "  run $r: no timings (request failed)"; continue; }
    pn=$(echo "$t" | jq -r '.prompt_n'); cn=$(echo "$t" | jq -r '.cached_n'); ps=$(echo "$t" | jq -r '.prompt_per_second')
    printf "  run %d: prompt_n=%s cached_n=%s  prefill=%.1f tok/s\n" "$r" "$pn" "$cn" "$ps"
    cold_tps="${cold_tps}${ps},"; cold_cached="${cold_cached}${cn},"
done

echo
echo "--- WARM (same prompt twice; second reuses the prefix) ---"
warm_tps=""; warm_cached=""
for r in $(seq 1 "$RUNS"); do
    # Prime (unique nonce so we measure THIS run's reuse, not the previous run's).
    pfx="warmset-$r preamble. "
    _=$(post_prompt "${pfx}${BASE_PROMPT}")
    t="$(post_prompt "${pfx}${BASE_PROMPT}")"   # identical → should reuse
    [ -z "$t" ] && { echo "  run $r: no timings (request failed)"; continue; }
    pn=$(echo "$t" | jq -r '.prompt_n'); cn=$(echo "$t" | jq -r '.cached_n'); ps=$(echo "$t" | jq -r '.prompt_per_second')
    printf "  run %d: prompt_n=%s cached_n=%s  prefill(suffix)=%.1f tok/s\n" "$r" "$pn" "$cn" "$ps"
    warm_tps="${warm_tps}${ps},"; warm_cached="${warm_cached}${cn},"
done

echo
echo "=== summary ($(basename "$MODEL")) ==="
echo "  COLD prefill: $(med "${cold_tps%,}") tok/s (median), cached_n median $(med "${cold_cached%,}")"
echo "  WARM prefill: $(med "${warm_tps%,}") tok/s (median), cached_n median $(med "${warm_cached%,}")"
echo "  (warm cached_n > 0 ⇒ prefix reuse engaged; warm suffix tok/s is tiny-N noise)"
