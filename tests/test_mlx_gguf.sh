#!/bin/bash
# Integration test for the MLX GGUF engine (lib/mlx-serve-gguf).
#
# Boots `mlx-serve` against every GGUF given (archs qwen35, qwen35moe, gemma4,
# IQ/K quants) and checks the whole path: routing to the MLX engine instead of
# llama.cpp, tokenizer + chat template rebuilt from the GGUF metadata, a correct
# greedy answer, streaming, PLD staying lossless (its verify steps run the
# multi-row matvec), and that `--engine llama` still hands the file to llama.cpp.
#
# Gated on fixtures so CI without a GGUF stays green:
#   MLX_GGUF_MODELS="/path/a.gguf /path/b.gguf" ./tests/test_mlx_gguf.sh [port]
#   MLX_GGUF_FLAGS="--skip-mem-preflight" for a model that barely fits.
set -uo pipefail

MODELS="${MLX_GGUF_MODELS:-${MLX_GGUF_MODEL:-}}"
PORT="${1:-8127}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0

if [ -z "$MODELS" ]; then
    echo -e "${YELLOW}SKIP${NC} test_mlx_gguf: set MLX_GGUF_MODELS=\"/path/to/a.gguf ...\" to run"
    exit 0
fi
if [ ! -f "$BIN" ]; then
    echo -e "${RED}ERROR${NC} $BIN not found — build with: zig build -Doptimize=ReleaseFast"
    exit 1
fi

ok()  { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "${2:-}" ] && echo "    $2"; }
assert_contains() { if grep -q "$2" <<< "$3"; then ok "$1"; else bad "$1" "missing '$2' in: $(echo "$3" | head -c 200)"; fi; }
assert_lacks()    { if grep -q "$2" <<< "$3"; then bad "$1" "unexpected '$2'"; else ok "$1"; fi; }

LOG="$(mktemp)"
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT

boot() {
    : > "$LOG"
    # shellcheck disable=SC2086
    "$BIN" --model "$MODEL" --serve --port "$PORT" --log-level info ${MLX_GGUF_OPT_IN---mlx-gguf} ${MLX_GGUF_FLAGS:-} "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 180); do
        if curl -fs --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"'; then return 0; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo -e "${RED}server died${NC}"; tail -20 "$LOG"; exit 1; fi
        sleep 1
    done
    echo -e "${RED}server never became healthy${NC}"; tail -20 "$LOG"; exit 1
}

stop() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }

# An edit task, the kind of request PLD drafts well on.
edit_task() {
    curl -fs --max-time 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Rename the variable total to result and return the full code, nothing else:\n\ndef add_all(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    print(total)\n    return total\n"}],"max_tokens":96,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

for MODEL in $MODELS; do
echo "→ starting mlx-serve on :$PORT with $MODEL"
boot

# 1. Served by the MLX engine, not an embedded one.
assert_contains "routes to the mlx gguf engine" "\[gguf\] engine: mlx" "$(cat "$LOG")"
assert_contains "weights come from the gguf"    "\[gguf\] mlx engine: .* tensors" "$(cat "$LOG")"
assert_lacks    "llama.cpp stays out of it"      "\[llama\] engine ready" "$(cat "$LOG")"

# 2. Tokenizer + template from GGUF metadata, kernels right: a greedy fact.
CHAT="$(curl -fs --max-time 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')"
assert_contains "greedy answer is right"      'Paris'             "$CHAT"
assert_contains "chat finishes cleanly"       '"finish_reason":"stop"' "$CHAT"
assert_lacks    "no template markers leak"    'im_end\|turn|'     "$CHAT"

# 3. A prompt wide enough for the tile mat-mat kernel (> 8 tokens) plus a longer
#    one for the dequantize + GEMM path (> 384 tokens) still answer sensibly.
LONG="$(python3 -c 'print("Ignore this filler. " * 120)')"
WIDE="$(curl -fs --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$LONG What is 2+2? Answer with the digit only.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")"
assert_contains "long prompt prefill answers" '4' "$(echo "$WIDE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"

# 4. Streaming.
STREAM="$(curl -fs --max-time 120 -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count to three"}],"max_tokens":16,"temperature":0,"stream":true}')"
assert_contains "stream emits chat.completion.chunk" 'chat.completion.chunk' "$STREAM"
assert_contains "stream terminates with [DONE]"      '\[DONE\]'              "$STREAM"

# 5. PLD (on by default) is lossless: same greedy text as --no-pld.
WITH_PLD="$(edit_task)"
stop
boot --no-pld
WITHOUT_PLD="$(edit_task)"
assert_contains "edit task answers" 'result' "$WITH_PLD"
if [ "$WITH_PLD" = "$WITHOUT_PLD" ]; then ok "PLD output equals --no-pld"; else bad "PLD output equals --no-pld" "$(diff <(echo "$WITH_PLD") <(echo "$WITHOUT_PLD") | head -5)"; fi

# 6. --engine llama turns the MLX engine off for the same file.
stop
boot --engine llama
assert_contains "--engine llama routes to llama.cpp" "\[llama\] engine ready" "$(cat "$LOG")"
assert_lacks    "--engine llama skips the mlx engine" "\[gguf\] engine: mlx" "$(cat "$LOG")"
stop

# 7. Without --mlx-gguf the engine is opt-in off: llama.cpp keeps the file.
MLX_GGUF_OPT_IN="" boot
assert_contains "no --mlx-gguf routes to llama.cpp" "\[llama\] engine ready" "$(cat "$LOG")"
assert_lacks    "no --mlx-gguf skips the mlx engine" "\[gguf\] engine: mlx" "$(cat "$LOG")"
stop
done

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
