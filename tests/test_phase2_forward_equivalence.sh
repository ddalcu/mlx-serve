#!/bin/bash
# test_phase2_forward_equivalence.sh — tripwire: the first N greedy tokens of the forward pass stay byte-identical to a
# baseline recorded by this script's own first run (tests/fixtures/phase2_forward_equivalence/). Covers a hybrid SSM
# (Qwen3.5-4B: GDN + chunked prefill) and plain attention (Gemma 4 E4B: sliding window).
#
#   QWEN_MLX_MODEL=<dir> GEMMA4_MLX_MODEL=<dir> ./tests/test_phase2_forward_equivalence.sh   # PORT (19099), BINARY
#
# PHASE2_REBASELINE=1 overwrites the fixture after an intentional change. Exit 1 = a case mismatched; a missing model
# skips its case.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

PORT="${PORT:-19099}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
QWEN="${QWEN_MLX_MODEL:-$HOME/.mlx-serve/models/lmstudio-community/Qwen3.5-4B-MLX-4bit}"
source "$(dirname "$0")/_lib_models.sh"
GEMMA="${GEMMA4_MLX_MODEL:-$(find_model mlx-community/gemma-4-e4b-it-4bit)}"
REBASELINE="${PHASE2_REBASELINE:-0}"
BASE="http://127.0.0.1:$PORT"

FIXDIR="tests/fixtures/phase2_forward_equivalence"
mkdir -p "$FIXDIR"

[ -x "$BIN" ] || { echo "fail: build mlx-serve first ($BIN)"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

SERVER_PID=""
LOG="$(mktemp)"
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
    rm -f "$LOG"
}
trap cleanup EXIT INT TERM

start_server() {
    local model="$1"
    pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
    sleep 1
    "$BIN" --model "$model" --serve --port "$PORT" --ctx-size 8192 \
        --prefix-cache-entries 4 --log-level warn > "$LOG" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 180 0.5 "$SERVER_PID" && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; return 1; }
    echo "fail: server didn't come up"
    tail -20 "$LOG"
    return 1
}

# Three prompts that exercise different sequence lengths and conv-state
# boundaries. Each is sent at temperature=0 with max_tokens=30 so we record
# 30 fully-deterministic tokens for the comparison.
gen_prompt() {
    case "$1" in
        long)
            python3 - <<'PY'
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Continue the story in exactly 4 sentences.\n\nBackground:\n" + para * 50, end="")
PY
            ;;
        short)
            printf '%s' "Write a single sentence describing a sunrise over the ocean. Do not use the word 'crimson'."
            ;;
        code)
            printf '%s' "Implement a Python function fib(n) that returns the n-th Fibonacci number using iteration. Provide only the code, no commentary."
            ;;
        *)
            echo "unknown prompt: $1" >&2; exit 2;;
    esac
}

# Run one (model, prompt) → write tokens csv to stdout. We hit the OAI chat
# API in non-streaming mode and decode the text + token id list. The token
# IDs come from `usage.completion_tokens_details` if the server emits them;
# otherwise we re-tokenize via the /v1/models tokenizer route. For mlx-serve
# we use the response text as the comparable artifact — it is the
# user-visible output and is what the byte-equivalence pin must hold.
run_case() {
    local prompt_kind="$1"
    local PROMPT
    PROMPT="$(gen_prompt "$prompt_kind")"
    local body
    body=$(jq -nc --arg p "$PROMPT" \
        '{messages:[{role:"user",content:$p}],max_tokens:30,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    curl -sf --max-time 90 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" \
        | jq -r '.choices[0].message.content'
}

EC=0
RESULTS=()

run_arch() {
    local arch_label="$1" model_dir="$2"
    if [ ! -d "$model_dir" ]; then
        echo "SKIP: $arch_label — model dir missing ($model_dir)"
        RESULTS+=("SKIP $arch_label")
        return 0
    fi
    echo "==> $arch_label  ($model_dir)"
    start_server "$model_dir" || { EC=1; RESULTS+=("FAIL $arch_label start"); return 0; }

    for prompt_kind in long short code; do
        local fixture="$FIXDIR/${arch_label}_${prompt_kind}.txt"
        local actual
        actual="$(run_case "$prompt_kind")"
        if [ -z "$actual" ]; then
            echo "  ❌ $prompt_kind: empty response"
            EC=1; RESULTS+=("FAIL $arch_label $prompt_kind empty")
            continue
        fi
        if [ "$REBASELINE" = "1" ] || [ ! -f "$fixture" ]; then
            printf '%s' "$actual" > "$fixture"
            echo "  📌 $prompt_kind: baseline recorded (${#actual} chars)"
            RESULTS+=("BASE $arch_label $prompt_kind")
            continue
        fi
        local expected
        expected="$(cat "$fixture")"
        if [ "$actual" = "$expected" ]; then
            echo "  ✅ $prompt_kind: byte-identical (${#actual} chars)"
            RESULTS+=("PASS $arch_label $prompt_kind")
        else
            echo "  ❌ $prompt_kind: DIVERGED"
            echo "  --- expected (${#expected} chars) ---"
            printf '%s\n' "$expected" | head -3
            echo "  --- actual (${#actual} chars) ---"
            printf '%s\n' "$actual" | head -3
            echo "  --- diff (first 20 lines) ---"
            diff <(printf '%s' "$expected") <(printf '%s' "$actual") | head -20
            EC=1; RESULTS+=("FAIL $arch_label $prompt_kind diverged")
        fi
    done
}

run_arch "qwen35-4b-mlx" "$QWEN"
run_arch "gemma4-e4b-mlx" "$GEMMA"

cleanup

echo
echo "=== summary ==="
for r in "${RESULTS[@]}"; do echo "  $r"; done
if [ "$EC" -ne 0 ]; then
    echo
    echo "FAIL: at least one case diverged from baseline."
    echo "If the behavior change is intentional, rebaseline with PHASE2_REBASELINE=1."
fi
exit "$EC"
