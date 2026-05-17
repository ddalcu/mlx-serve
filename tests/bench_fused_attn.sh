#!/bin/bash
# Fused quant-attention bench (Plan ricky Phase 2).
#
# Sweeps (mode × context) for a single model, emitting CSV with cold
# prefill ms + warm decode tok/s. Crossover decision (where fused beats
# dense) is data-driven — this script produces the data; the v2 follow-up
# will read the CSV and pick a default-flip policy.
#
# Defaults bench: gemma-4-e4b-it-4bit at 1k / 4k contexts, dense vs fused
# attention mode, --kv-quant 4. Override via positional args / env vars.
#
# Usage:
#   ./tests/bench_fused_attn.sh [model_dir] [port] [out.csv]
#   CTX_LENS=1024,4096,16384 ./tests/bench_fused_attn.sh

set -uo pipefail

MODEL_DIR="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-19033}"
OUT_CSV="${3:-/tmp/bench_fused_attn.csv}"
CTX_LENS="${CTX_LENS:-1024,4096}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
[[ -x "$BINARY" ]] || { echo "Build first: zig build -Doptimize=ReleaseFast" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Model not found: $MODEL_DIR" >&2; exit 1; }

trap 'pkill -9 -x mlx-serve 2>/dev/null; true' EXIT

build_prompt() {
    # Build a deterministic ~$1-token prompt (approximate, each "TQBFJ" repeat
    # is ~10 tokens for Gemma's tokenizer).
    local target=$1
    local reps=$((target / 10))
    python3 -c "print('The quick brown fox jumps over the lazy dog. ' * $reps)"
}

run_one() {
    # Args: ctx, mode (dense|fused). Returns one CSV line.
    local ctx="$1" mode="$2"
    local model_name
    model_name=$(basename "$MODEL_DIR")

    pkill -9 -x mlx-serve 2>/dev/null
    sleep 1
    local logfile=/tmp/bench_fused_engine.log
    "$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" \
        --kv-quant 4 --kv-attn-mode "$mode" \
        --no-pld --ctx-size $((ctx + 512)) \
        --log-level info > "$logfile" 2>&1 &
    local pid=$!
    local i
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 0.5
        kill -0 "$pid" 2>/dev/null || { echo "$model_name,$ctx,$mode,ERROR_SERVER_DIED,," ; return; }
    done

    local prompt
    prompt=$(build_prompt "$ctx")
    local body
    body=$(jq -nc --arg p "$prompt" --argjson mt "$DECODE_TOKENS" \
        '{messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0.0,stream:false,enable_thinking:false}')

    # Cold first request — measures prefill + decode end-to-end.
    local t0 t1
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -fs -m 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body" >/dev/null
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    local cold_ms=$((t1 - t0))

    # Server log carries the authoritative prefill/decode tok/s.
    local prefill_tps decode_tps
    local last_line
    last_line=$(grep -oE '\[prefill:[^]]*\] \[[^]]*\]' "$logfile" | tail -1)
    prefill_tps=$(echo "$last_line" | grep -oE 'prefill: [0-9.]+' | grep -oE '[0-9.]+$' || echo "0")
    decode_tps=$(echo "$last_line" | grep -oE 'decode: [0-9.]+' | grep -oE '[0-9.]+$' || echo "0")

    pkill -9 -x mlx-serve 2>/dev/null
    sleep 1
    echo "$model_name,$ctx,$mode,$cold_ms,$prefill_tps,$decode_tps"
}

echo "model,ctx_tokens,mode,cold_ms,prefill_tps,decode_tps" > "$OUT_CSV"
echo "=== bench_fused_attn.sh ==="
echo "  model    : $(basename "$MODEL_DIR")"
echo "  contexts : $CTX_LENS"
echo "  modes    : dense, fused"
echo "  decode   : $DECODE_TOKENS tokens"
echo "  out      : $OUT_CSV"
echo

IFS=',' read -ra CTX_ARR <<< "$CTX_LENS"
for ctx in "${CTX_ARR[@]}"; do
    for mode in dense fused; do
        line=$(run_one "$ctx" "$mode")
        echo "  $line"
        echo "$line" >> "$OUT_CSV"
    done
done

echo
echo "=== summary ==="
column -t -s',' "$OUT_CSV" | sed 's/^/  /'
