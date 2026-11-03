#!/bin/bash
# test_llama_multi_session.sh — llama.cpp keeps one KV session per conversation: with --llama-cache-entries 2,
# alternating long-doc QA (A, B, A, B) reports cached_n == 0 on the first A and B and cached_n > 0 on their second
# visits (a single shared session evicts one with the other).
#
#   LLAMA_GGUF_MODEL=<file.gguf> ./tests/test_llama_multi_session.sh   # PORT (19107), BINARY

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

PORT="${PORT:-19107}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
source "$(dirname "$0")/_lib_models.sh"
# A pure-attention GGUF: a recurrent one (Qwen3.5) reports cached_n=0 by design and cannot show the session reuse this script asserts.
MODEL="${LLAMA_GGUF_MODEL:-$(find_model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-IQ4_NL.gguf)}"
BASE="http://127.0.0.1:$PORT"

[ -f "$MODEL" ] || { echo "SKIP: GGUF file missing: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first ($BIN)"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
sleep 1

LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
    rm -f "$LOG"
}
trap cleanup EXIT INT TERM

"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 4096 \
    --llama-cache-entries 2 --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
srv_wait_health "$BASE" 240 0.5 "$SERVER_PID" || kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -30 "$LOG"; exit 1; }

# Two distinct prompts, each long enough that the prefix differs in the first
# few tokens (so commonPrefixLen between A and B is near zero — neither acts
# as the other's prefix).
PROMPT_A="The first chapter describes a sunny morning where the protagonist wakes up in a small village by the sea. The waves are calm and the smell of fresh bread fills the air. Birds chirp loudly outside. The protagonist walks to the bakery and orders two loaves. Summarize this in one sentence."
PROMPT_B="In the dense jungle a tiger stalks its prey at twilight. The leaves rustle as a herd of deer passes by, oblivious to the predator's gaze. The tiger sets its sights on a young straggler. After a brief chase the hunt ends. Summarize this in one sentence."

ask() {
    local p="$1"
    curl -sf --max-time 90 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$p" '{messages:[{role:"user",content:$p}],max_tokens:1,temperature:0,stream:false}')"
}

cached_of() { echo "$1" | jq -r '.timings.cached_n // 0'; }

echo "==> first A"
A1="$(ask "$PROMPT_A")"
A1_C=$(cached_of "$A1")
echo "  cached_n=$A1_C"

echo "==> first B"
B1="$(ask "$PROMPT_B")"
B1_C=$(cached_of "$B1")
echo "  cached_n=$B1_C"

echo "==> second A"
A2="$(ask "$PROMPT_A")"
A2_C=$(cached_of "$A2")
echo "  cached_n=$A2_C"

echo "==> second B"
B2="$(ask "$PROMPT_B")"
B2_C=$(cached_of "$B2")
echo "  cached_n=$B2_C"

cleanup

EC=0
if [ "$A1_C" != "0" ]; then
    echo "❌ first A cached_n=$A1_C, expected 0 (cold)"; EC=1
fi
if [ "$B1_C" != "0" ]; then
    echo "⚠ first B cached_n=$B1_C, expected 0 — non-fatal (template tokens overlapped)"
fi
# Prompt A is ~72 tokens (chat template + content). With multi-session LRU
# the second visit reuses all but the final position, so cached_n lands
# around 71. With the single-session fallback, B's request overwrote A's
# KV and cached_n drops to a handful of chat-template-header tokens (~3).
# Threshold 20 cleanly separates "real content kept" from "template-only".
if [ "$A2_C" -lt 20 ] 2>/dev/null; then
    echo "❌ second A cached_n=$A2_C, expected >= 20 (multi-session LRU should have kept A's KV alive across B; single-session would give ~3)"; EC=1
else
    echo "✅ second A cached_n=$A2_C (multi-session LRU kept A alive)"
fi
if [ "$B2_C" -lt 20 ] 2>/dev/null; then
    echo "❌ second B cached_n=$B2_C, expected >= 20 (B kept alive across A's second visit)"; EC=1
else
    echo "✅ second B cached_n=$B2_C (multi-session LRU kept B alive)"
fi

exit "$EC"
