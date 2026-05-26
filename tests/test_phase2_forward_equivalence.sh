#!/bin/bash
# test_phase2_forward_equivalence.sh — Phase 2 of performance-plan.md.
#
# Pins the first N greedy temp=0 tokens of mlx-serve's forward pass against
# a baseline captured on main. Any change to the forward path (compileForward
# wiring, op-fusion, kernel rework, dtype tweak, ...) must keep this test
# byte-identical. The test is a TRIPWIRE — its own first run records the
# baseline; subsequent runs compare against it. To re-baseline (intentional
# behavior change), delete the cached fixtures under
# tests/fixtures/phase2_forward_equivalence/ and rerun.
#
# Coverage:
#   - Hybrid SSM (Qwen3.5-4B-MLX-4bit) — exercises gatedDeltaNet + the SSM
#     scan kernel + chunked-prefill (forward dispatched per chunk).
#   - Plain attention (Gemma 4 E4B 4-bit) — exercises forwardStandard +
#     sliding-window attention.
# Add MoE / DSV4 fixtures as they become available.
#
# Env:
#   QWEN_MLX_MODEL     Path to Qwen3.5-4B MLX dir (or compatible hybrid). If
#                      unset / missing, that case is skipped (test still
#                      records pass/skip).
#   GEMMA4_MLX_MODEL   Path to Gemma 4 E4B 4-bit dir. Skipped if unset.
#   PORT               Server port. Default 19099.
#   PHASE2_REBASELINE  If set to "1", overwrite the fixture instead of
#                      comparing. Use after an intentional behavior change.
#   BINARY             Path to mlx-serve. Default ./zig-out/bin/mlx-serve.
#
# Exit codes:
#   0  all pinned cases byte-identical (or skipped because model missing)
#   1  at least one case mismatched
#
# Outputs fixture content under tests/fixtures/phase2_forward_equivalence/

set -uo pipefail

PORT="${PORT:-19099}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
QWEN="${QWEN_MLX_MODEL:-/Volumes/Sandisk_1TB/Models/mlx-community/Qwen3.5-4B-MLX-4bit}"
GEMMA="${GEMMA4_MLX_MODEL:-/Users/david/.mlx-serve/models/gemma-4-e4b-it-4bit}"
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
    for _ in $(seq 1 180); do
        curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; return 1; }
        sleep 0.5
    done
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
