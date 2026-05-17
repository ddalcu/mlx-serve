#!/bin/bash
# sweep_agent_memory.sh — run the 11-turn agent memory test against every
# locally-available architecture. For each arch:
#   1. Boot mlx-serve on a free port.
#   2. Invoke tests/test_long_agent_memory.py against it.
#   3. Count PASS/FAIL lines and the final "tests passed" line.
#   4. Tear the server down.
#
# Output: AGENT_MEMORY_RESULTS.md (one row per arch).
#
# Override which archs are tested via SWEEP_MODELS=<csv of logical names>.

set -uo pipefail

cd "$(dirname "$0")/.."

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11296}"
RESULTS="${RESULTS:-AGENT_MEMORY_RESULTS.md}"

# Mirror sweep_all_archs.sh's table — pipe-separated entries.
MODELS=(
    "gemma4-e2b-4bit|Gemma 4 E2B (4-bit)|$HOME/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit|mlx"
    "gemma4-e4b-4bit|Gemma 4 E4B (4-bit)|$HOME/.lmstudio/models/mlx-community/gemma-4-e4b-it-4bit|mlx"
    "gemma4-26b-moe-4bit|Gemma 4 26B-A4B MoE (4-bit)|$HOME/.mlx-serve/models/gemma-4-26b-a4b-it-4bit|mlx"
    "gemma4-31b-4bit|Gemma 4 31B (4-bit)|$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit|mlx"
    "qwen36-27b-4bit|Qwen 3.6 27B dense (4-bit)|$HOME/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit|mlx"
    "qwen36-35b-moe-ud|Qwen 3.6 35B-A3B MoE UD (4-bit)|$HOME/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit|mlx"
    "dsv4-flash-gguf|DeepSeek-V4-Flash (GGUF, ds4)|$HOME/projects/agents/ds4/ds4flash.gguf|gguf"
)

if [[ -n "${SWEEP_MODELS:-}" ]]; then
    IFS=',' read -r -a WANTED <<< "$SWEEP_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do
            if [[ "$name" == "$w" ]]; then FILTERED+=("$entry"); fi
        done
    done
    MODELS=("${FILTERED[@]}")
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fatal] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

pkill -f 'mlx-serve --serve' >/dev/null 2>&1 || true
sleep 1

echo "# 11-turn agent memory sweep — $(date '+%Y-%m-%d %H:%M')" > "$RESULTS"
echo "" >> "$RESULTS"
echo "Binary: \`$BINARY\` ($(du -h "$BINARY" | awk '{print $1}'))" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "Each cell runs \`tests/test_long_agent_memory.py\` (11 turns: plant facts → tool call → recall codename → thinking+tool → recall language → recall deadline → multi-turn agent loop → summary check)." >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| Architecture | Result | Pass / Total | Failures | Time |" >> "$RESULTS"
echo "|---|---|---|---|---|" >> "$RESULTS"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'

wait_for_health() {
    local port="$1" timeout="${2:-180}" pid="$3"
    for ((i=1; i<=timeout; i++)); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        sleep 1
    done
    return 1
}

run_one() {
    local name="$1" display="$2" path="$3" type="$4"
    echo -e "${BLUE}=== $display ===${NC}"

    if [[ "$type" == "mlx" && ! -d "$path" ]] || [[ "$type" == "gguf" && ! -f "$path" ]]; then
        echo -e "${YELLOW}[skip]${NC} not found: $path"
        echo "| $display | skip (missing) | — | — | — |" >> "$RESULTS"
        return 0
    fi

    local log
    log=$(mktemp)
    echo "  booting on port $PORT (log: $log)…"
    "$BINARY" --model "$path" --serve --port "$PORT" --log-level info > "$log" 2>&1 &
    local sp=$!
    if ! wait_for_health "$PORT" 240 "$sp"; then
        echo -e "${RED}[fail]${NC} server didn't come up"
        tail -20 "$log" | sed 's/^/    /'
        kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
        echo "| $display | FAIL (boot) | — | — | — |" >> "$RESULTS"
        return 1
    fi
    echo "  up. running 11-turn agent memory test…"

    local t0 t1 dt out total_pass total_fail status fails_csv
    t0=$(date +%s)
    out=$(python3 tests/test_long_agent_memory.py "$PORT" 2>&1 || true)
    t1=$(date +%s)
    dt=$((t1 - t0))

    # Strip ANSI escapes once — the test script colors its output, but the
    # raw bytes after PASS/FAIL are the color-reset sequence (not whitespace),
    # which would defeat a `PASS[[:space:]]` regex.
    local clean
    clean=$(printf '%s' "$out" | sed -E 's/\x1b\[[0-9;]*[mK]//g')
    # Prefer the final summary line ("Assertions: N (X pass, Y fail)") when
    # present; otherwise fall back to per-line PASS/FAIL counts.
    if printf '%s' "$clean" | grep -qE 'Assertions:[[:space:]]+[0-9]+'; then
        local summary
        summary=$(printf '%s' "$clean" | grep -E 'Assertions:[[:space:]]+[0-9]+' | tail -1)
        total_pass=$(printf '%s' "$summary" | grep -oE '[0-9]+ pass' | head -1 | awk '{print $1}')
        total_fail=$(printf '%s' "$summary" | grep -oE '[0-9]+ fail' | head -1 | awk '{print $1}')
        : "${total_pass:=0}"
        : "${total_fail:=0}"
    else
        total_pass=$(printf '%s' "$clean" | grep -cE '^[[:space:]]*PASS[[:space:]]' || true)
        total_fail=$(printf '%s' "$clean" | grep -cE '^[[:space:]]*FAIL[[:space:]]' || true)
    fi
    local total=$((total_pass + total_fail))

    # Capture distinct failure descriptions (one-line each).
    fails_csv=$(printf '%s' "$clean" | grep -E '^[[:space:]]*FAIL[[:space:]]' | sed -E 's/^[[:space:]]*FAIL[[:space:]]+//' | head -3 | tr '\n' ';' | sed 's/;$//')
    if [[ -z "$fails_csv" ]]; then fails_csv="—"; fi

    # Final summary line says e.g. "2/40 assertions failed" → derive status
    if [[ $total_fail -eq 0 && $total_pass -gt 0 ]]; then
        status="${GREEN}PASS${NC}"
        status_md="PASS"
        echo -e "  $status   $total_pass/$total assertions, ${dt}s"
    else
        status="${RED}FAIL${NC}"
        status_md="FAIL"
        echo -e "  $status   $total_pass/$total assertions, ${dt}s"
        echo "  failures: $fails_csv" | head -c 200; echo
    fi

    # Save full log so we can dig into failures.
    local out_log
    out_log="/tmp/agent_memory_${name}.log"
    printf '%s\n' "$out" > "$out_log"
    echo "  full log: $out_log"

    echo "| $display | $status_md | $total_pass / $total | $fails_csv | ${dt}s |" >> "$RESULTS"

    kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
    rm -f "$log"
    return 0
}

trap 'pkill -f "mlx-serve --serve" 2>/dev/null || true' EXIT

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name display path type <<< "$entry"
    run_one "$name" "$display" "$path" "$type" || true
    pkill -f "mlx-serve --serve" 2>/dev/null || true
    sleep 3
done

echo
echo "=== Agent-memory sweep complete ==="
echo "Results: $RESULTS"
echo
cat "$RESULTS"
