#!/bin/bash
# test_validator_matrix.sh — API-compliance + agentic matrix across every
# locally-available Zig MLX architecture plus one llama-engine GGUF and one
# ds4-engine GGUF.
#
# Per model, two layers:
#   1. llmprobe (~/projects/agents/responses-chat-messages-validator) in auto
#      mode — compliance suites for /v1/responses, /v1/chat/completions and
#      /v1/messages against a server this script boots and tears down.
#   2. The pi 2-turn agentic html case (multi-turn tool calls: create
#      mlx.html, then add JS) via tests/pi_integration_run.sh, which manages
#      its own server lifecycle and appends audit_format leak markers.
#
# One model loaded at a time; missing weights skip cleanly. The 81GB ds4
# model cannot coexist with another large resident model — this script
# pkills ALL serving mlx-serve instances up front (including the app's).
#
# Usage:
#   ./tests/test_validator_matrix.sh                      # full matrix
#   VALIDATOR_MODELS=gemma4,ds4 ./tests/...               # csv filter
#   SKIP_PI=1 ./tests/test_validator_matrix.sh            # llmprobe only
#   SKIP_PROBE=1 ./tests/test_validator_matrix.sh         # pi only
#   BINARY=...  PORT=...  LLMPROBE_DIR=...                # overrides
#
# Output: per-model logs in tests/validator-results/, summary table on
# stdout + tests/validator-results/summary.tsv. Exit 1 if anything failed.

set -uo pipefail
cd "$(dirname "$0")/.."

REPO="$(pwd)"
BINARY="${BINARY:-$REPO/zig-out/bin/mlx-serve}"
PORT="${PORT:-11298}"
LLMPROBE_DIR="${LLMPROBE_DIR:-$HOME/projects/agents/responses-chat-messages-validator}"
LLMPROBE_MJS="$LLMPROBE_DIR/bin/dist/llmprobe.mjs"
RESULTS="$REPO/tests/validator-results"
SUMMARY="$RESULTS/summary.tsv"
mkdir -p "$RESULTS"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# logical|display|path|pi_case|probe_timeout_s
# pi_case must exist in pi_integration_run.sh's html matrix.
MODELS=(
    "qwen2|Qwen2.5-Coder 32B 8-bit (qwen2)|$HOME/.mlx-serve/models/mlx-community/Qwen2.5-Coder-32B-Instruct-8bit|html-coder|150"
    "gemma4|Gemma 4 E4B 8-bit (gemma4)|$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit|html-e4b|90"
    "gemma3|Gemma 3 12B QAT 4-bit (gemma3)|$HOME/.mlx-serve/models/mlx-community/gemma-3-12b-it-qat-4bit|html-gemma3|90"
    "qwen36|Qwen3.6 27B dense 4-bit (qwen3_5)|$HOME/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit|html-qwen36|150"
    "qwen36-moe|Qwen3.6 35B-A3B 6-bit (qwen3_5_moe)|$HOME/.mlx-serve/models/Qwen3.6-35B-A3B-6bit|html-qwen-think|150"
    "qwen3-coder|Qwen3-Coder 30B-A3B 8-bit (qwen3_moe)|$HOME/.mlx-serve/models/mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit|html-coder|150"
    "e4b-gguf|Gemma 4 E4B GGUF (llama engine)|$HOME/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf|html-e4b-gguf|150"
    "ds4|DeepSeek-V4-Flash GGUF (ds4 engine)|$HOME/.mlx-serve/models/antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf|html-ds4|240"
)

if [[ -n "${VALIDATOR_MODELS:-}" ]]; then
    IFS=',' read -r -a WANTED <<< "$VALIDATOR_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do
            [[ "$name" == "$w" ]] && FILTERED+=("$entry")
        done
    done
    MODELS=("${FILTERED[@]:-}")
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fatal] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if [[ -z "${SKIP_PROBE:-}" && ! -f "$LLMPROBE_MJS" ]]; then
    echo "[fatal] llmprobe bundle missing: $LLMPROBE_MJS (cd $LLMPROBE_DIR && npm run build:cli)"
    exit 1
fi

kill_servers() {
    pkill -f 'mlx-serve.*--serve' 2>/dev/null || true
    for _ in $(seq 1 10); do
        pgrep -f 'mlx-serve.*--serve' >/dev/null || return 0
        sleep 0.5
    done
}

start_server() { # path logfile -> 0 on healthy
    "$BINARY" --model "$1" --serve --port "$PORT" --log-level info \
        --ctx-size 32768 > "$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    return 1
}

strip_ansi() { sed -E $'s/\x1b\\[[0-9;]*m//g'; }

trap 'kill_servers' EXIT
kill_servers

[[ -f "$SUMMARY" ]] || printf "timestamp\tmodel\tprobe\tpi\tnotes\n" > "$SUMMARY"

overall_fail=0
declare -a ROWS

for entry in "${MODELS[@]}"; do
    IFS='|' read -r logical display path pi_case probe_timeout <<< "$entry"
    echo -e "\n${YELLOW}===== $logical — $display =====${NC}"
    if [[ ! -e "$path" ]]; then
        echo -e "${YELLOW}SKIP${NC}: model path missing: $path"
        ROWS+=("$logical|SKIP|SKIP|model missing")
        continue
    fi

    probe_cell="skipped"; pi_cell="skipped"; notes=""

    # ---- Layer 1: llmprobe across all three API surfaces -------------------
    if [[ -z "${SKIP_PROBE:-}" ]]; then
        server_log="$RESULTS/$logical.server.log"
        probe_log="$RESULTS/$logical.probe.log"
        if start_server "$path" "$server_log"; then
            node "$LLMPROBE_MJS" "127.0.0.1:$PORT" --timeout "$probe_timeout" \
                2>&1 | strip_ansi > "$probe_log"
            probe_rc=${PIPESTATUS[0]}
            # Per-suite lines: "Results: X passed, Y failed, Z skipped, N total"
            probe_cell=$(awk '/^Results:/ {gsub(/[^0-9 ]/,""); p+=$1; f+=$2; s+=$3; t+=$4}
                END { if (t=="") print "no-results"; else printf "%d/%d", p, t-s }' "$probe_log")
            failed_ids=$(grep '^Failed:' "$probe_log" | sed 's/^Failed: //' | paste -sd';' - | head -c 300)
            if [[ "$probe_rc" -ne 0 || "$probe_cell" == "no-results" ]]; then
                overall_fail=1
                notes="$notes probe-fail:[${failed_ids:-see $probe_log}]"
                echo -e "${RED}probe: $probe_cell${NC} ($probe_log)"
            else
                echo -e "${GREEN}probe: $probe_cell${NC}"
            fi
        else
            overall_fail=1
            probe_cell="server-start-fail"
            notes="$notes server-start-fail"
            echo -e "${RED}server failed to boot — tail of $server_log:${NC}"
            tail -n 20 "$server_log"
        fi
        kill_servers
    fi

    # ---- Layer 2: pi agentic 2-turn html case (multi-turn tool calls) ------
    if [[ -z "${SKIP_PI:-}" ]]; then
        pi_log="$RESULTS/$logical.pi.log"
        MLX_BIN="$BINARY" PI_CASES="$pi_case" "$REPO/tests/pi_integration_run.sh" html \
            2>&1 | strip_ansi > "$pi_log"
        score_line=$(grep -E '^SCORE:' "$pi_log" | tail -n1)
        if [[ -n "$score_line" ]]; then
            pi_cell=$(echo "$score_line" | awk '{print $2}')
            pi_notes=$(echo "$score_line" | cut -d' ' -f3-)
            notes="$notes $pi_notes"
            if echo "$score_line" | grep -qE ' -[a-z]' || [[ "${pi_cell%%/*}" != "${pi_cell##*/}" && "${pi_cell%%/*}" -lt "${pi_cell##*/}" ]]; then
                overall_fail=1
                echo -e "${RED}pi: $score_line${NC} ($pi_log)"
            else
                echo -e "${GREEN}pi: $score_line${NC}"
            fi
        else
            overall_fail=1
            pi_cell="no-score"
            notes="$notes pi-no-score"
            echo -e "${RED}pi produced no SCORE line — tail of $pi_log:${NC}"
            tail -n 20 "$pi_log"
        fi
    fi

    ROWS+=("$logical|$probe_cell|$pi_cell|${notes# }")
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$logical" \
        "$probe_cell" "$pi_cell" "${notes# }" >> "$SUMMARY"
done

echo ""
echo "===== validator matrix summary ====="
printf "%-14s %-18s %-10s %s\n" "model" "llmprobe" "pi" "notes"
for row in "${ROWS[@]}"; do
    IFS='|' read -r m p a n <<< "$row"
    printf "%-14s %-18s %-10s %s\n" "$m" "$p" "$a" "$n"
done
echo ""
[[ "$overall_fail" -eq 0 ]] && echo -e "${GREEN}ALL GREEN${NC}" || echo -e "${RED}FAILURES — see $RESULTS${NC}"
exit "$overall_fail"
