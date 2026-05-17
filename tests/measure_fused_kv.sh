#!/bin/bash
# measure_fused_kv.sh — re-run the bench-prompt decode cell on Gemma 4 31B
# and Qwen 3.6 27B with --kv-attn-mode {dense,fused} and report side-by-side
# tok/s. Targets the two archs where mlx-serve currently ties or loses to
# mlx-lm; the hypothesis is that the fused quantized-attention path closes
# (or flips) the gap.
#
# Output: FUSED_KV_RESULTS.md.

set -uo pipefail

cd "$(dirname "$0")/.."

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11297}"
MAX_TOKENS="${MAX_TOKENS:-128}"
RESULTS="${RESULTS:-FUSED_KV_RESULTS.md}"
BENCH_PROMPT="${BENCH_PROMPT:-You are a careful technical writer. Write an 80-word vivid description of a thunderstorm over the Pacific Ocean from the perspective of a sailor. Include sensory details: smell of ozone, sound of thunder, rhythm of waves, contrast of dark clouds against silver lightning. Avoid clichés. Begin directly with the description.}"

# Note: --kv-attn-mode fused is only effective at --kv-quant 4 or 8 per
# CLAUDE.md ("fused: opt-in; only effective at --kv-quant 4 or 8"). We test
# the matrix dense+off (baseline), dense+8 (KV quant alone), fused+8 (full
# combo) so we can attribute any win between KV quant and the fused path.
MODELS=(
    "gemma4-31b-4bit|Gemma 4 31B (4-bit weights)|$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit"
    "qwen36-27b-4bit|Qwen 3.6 27B dense (4-bit weights)|$HOME/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit"
)

CONFIGS=(
    "dense+off|--kv-quant off|"
    "dense+8|--kv-quant 8|--kv-attn-mode dense"
    "fused+8|--kv-quant 8|--kv-attn-mode fused"
)

if [[ ! -x "$BINARY" ]]; then
    echo "[fatal] binary missing: $BINARY"; exit 1
fi
pkill -f 'mlx-serve --serve' >/dev/null 2>&1 || true
sleep 1

echo "# Fused KV-attention measurement — $(date '+%Y-%m-%d %H:%M')" > "$RESULTS"
echo "" >> "$RESULTS"
echo "| Architecture | Config | prefill tok/s | decode tok/s |" >> "$RESULTS"
echo "|---|---|---|---|" >> "$RESULTS"

wait_for_health() {
    local port="$1" timeout="${2:-180}" pid="$3"
    for ((i=1; i<=timeout; i++)); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        sleep 1
    done
    return 1
}

run_cell() {
    local model_path="$1" display="$2" config_label="$3" kv_quant_arg="$4" kv_attn_arg="$5"
    local log
    log=$(mktemp)
    echo "  → $config_label"
    # shellcheck disable=SC2206  # intentional word-splitting
    local extra=( $kv_quant_arg $kv_attn_arg )
    "$BINARY" --model "$model_path" --serve --port "$PORT" --max-tokens "$MAX_TOKENS" "${extra[@]}" > "$log" 2>&1 &
    local sp=$!
    if ! wait_for_health "$PORT" 180 "$sp"; then
        echo "    BOOT FAIL — tail:"
        tail -10 "$log" | sed 's/^/      /'
        kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
        echo "| $display | $config_label | BOOT FAIL | BOOT FAIL |" >> "$RESULTS"
        rm -f "$log"
        return 1
    fi

    # Warmup, then the bench prompt.
    local payload
    payload=$(BENCH_PROMPT="$BENCH_PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import json,os; print(json.dumps({"model":"mlx-serve","messages":[{"role":"user","content":os.environ["BENCH_PROMPT"]}],"max_tokens":int(os.environ["MAX_TOKENS"]),"temperature":0.0}))')
    # Warm.
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1 || true
    local log_before
    log_before=$(wc -c <"$log")
    # Measured run.
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1 || true
    sleep 0.4
    local slice prefill_tps decode_tps
    slice=$(tail -c +$((log_before + 1)) "$log")
    prefill_tps=$(printf '%s' "$slice" | grep -oE 'prefill: [0-9.]+ tok/s' | tail -1 | awk '{print $2}')
    decode_tps=$(printf '%s' "$slice" | grep -oE 'decode: [0-9.]+ tok/s' | tail -1 | awk '{print $2}')
    : "${prefill_tps:=—}"; : "${decode_tps:=—}"
    echo "    prefill=$prefill_tps  decode=$decode_tps"
    echo "| $display | $config_label | $prefill_tps | $decode_tps |" >> "$RESULTS"

    kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
    rm -f "$log"
    sleep 2
}

trap 'pkill -f "mlx-serve --serve" 2>/dev/null || true' EXIT

for m in "${MODELS[@]}"; do
    IFS='|' read -r name display path <<< "$m"
    if [[ ! -d "$path" ]]; then
        echo "[skip] $display (missing $path)"
        continue
    fi
    echo "=== $display ==="
    for c in "${CONFIGS[@]}"; do
        IFS='|' read -r clabel kvq kva <<< "$c"
        run_cell "$path" "$display" "$clabel" "$kvq" "$kva" || true
    done
done

echo
echo "=== Done ==="
cat "$RESULTS"
