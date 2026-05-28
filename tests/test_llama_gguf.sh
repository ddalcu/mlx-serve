#!/bin/bash
# Integration test for the embedded llama.cpp GGUF engine.
#
# Boots `mlx-serve` against a generic (non-DeepSeek-V4-Flash) .gguf and verifies
# the whole llama path end-to-end: auto-routing to the llama engine, /health,
# /v1/models, chat completions (non-streaming + streaming), and the Anthropic
# Messages API — all served through libllama via mlx-serve's own HTTP layer.
#
# Gated on a fixture so CI without a GGUF stays green:
#   LLAMA_GGUF_MODEL=/path/to/model.gguf ./tests/test_llama_gguf.sh [port]
# A small instruct model works well, e.g. Qwen2.5-0.5B-Instruct Q4_K_M.
set -uo pipefail

MODEL="${LLAMA_GGUF_MODEL:-}"
PORT="${1:-8123}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0

if [ -z "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_llama_gguf: set LLAMA_GGUF_MODEL=/path/to/model.gguf to run"
    exit 0
fi
if [ ! -f "$BIN" ]; then
    echo -e "${RED}ERROR${NC} $BIN not found — build with: zig build -Doptimize=ReleaseFast"
    exit 1
fi

ok()  { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "${2:-}" ] && echo "    $2"; }
assert_contains() { if echo "$3" | grep -q "$2"; then ok "$1"; else bad "$1" "missing '$2' in: $(echo "$3" | head -c 200)"; fi; }

LOG="$(mktemp)"
echo "→ starting mlx-serve on :$PORT with $MODEL"
"$BIN" --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT

# Wait for health (model load + Metal warmup can take a few seconds).
for i in $(seq 1 30); do
    if curl -fs --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"'; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo -e "${RED}server died${NC}"; tail -20 "$LOG"; exit 1; fi
    sleep 1
done

# 1. Routed to the llama engine (not ds4, not MLX).
assert_contains "routes to embedded llama engine" "\[llama\] engine ready" "$(cat "$LOG")"

# 2. /v1/models reports a ready gguf model.
MODELS="$(curl -fs --max-time 5 "$BASE/v1/models")"
assert_contains "/v1/models lists the model"        '"object":"model"'        "$MODELS"
assert_contains "/v1/models reports gguf arch"      '"architecture":"gguf"'   "$MODELS"
assert_contains "/v1/models marks it ready"         '"state":"ready"'         "$MODELS"

# 3. Non-streaming chat completion produces content + a clean stop.
CHAT="$(curl -fs --max-time 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with exactly: hello"}],"max_tokens":16,"temperature":0}')"
assert_contains "chat returns assistant content" '"content"'      "$CHAT"
assert_contains "chat finishes (stop/length)"    'finish_reason'  "$CHAT"
assert_contains "chat reports usage"             'completion_tokens' "$CHAT"

# 4. Streaming chat emits SSE deltas and a terminator.
STREAM="$(curl -fs --max-time 60 -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count to three"}],"max_tokens":16,"temperature":0,"stream":true}')"
assert_contains "stream emits chat.completion.chunk" 'chat.completion.chunk' "$STREAM"
assert_contains "stream terminates with [DONE]"      '\[DONE\]'              "$STREAM"

# 5. Anthropic Messages API over the same engine.
ANTHROPIC="$(curl -fs --max-time 60 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d '{"model":"x","max_tokens":16,"messages":[{"role":"user","content":"Say hi"}]}')"
assert_contains "/v1/messages returns text block" '"type":"text"' "$ANTHROPIC"
assert_contains "/v1/messages sets stop_reason"   'stop_reason'   "$ANTHROPIC"

# 6. Prompt-prefix KV reuse (persistent session). Two requests sharing a long
#    prefix: the second must reuse it (timings.cached_n > 0) instead of paying a
#    full cold prefill again — the core Phase 3 win and the regression guard for
#    the persistent-session plumbing (server → scheduler → session → timings).
PREFIX="In a faraway land there lived a wise old engineer who loved fast servers. "
PREFIX="$PREFIX$PREFIX$PREFIX$PREFIX$PREFIX$PREFIX"  # ~90 tokens of shared context
cached_n_of() { echo "$1" | grep -o '"cached_n":[0-9]*' | grep -o '[0-9]*' | head -1; }
REQ1="$(curl -fs --max-time 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${PREFIX}Tell me about apples.\"}],\"max_tokens\":8,\"temperature\":0}")"
REQ2="$(curl -fs --max-time 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${PREFIX}Tell me about oranges.\"}],\"max_tokens\":8,\"temperature\":0}")"
assert_contains "chat response exposes prefill timings" '"cached_n"' "$REQ2"
C1="$(cached_n_of "$REQ1")"; C2="$(cached_n_of "$REQ2")"
if [ -n "${C2:-}" ] && [ "${C2:-0}" -gt 0 ]; then
    ok "prefix reuse: req1 cached_n=${C1:-?}, req2 cached_n=${C2} (> 0)"
else
    bad "prefix reuse (req2 should reuse the shared prefix)" "req1 cached_n=${C1:-?}, req2 cached_n=${C2:-?}; req2=$(echo "$REQ2" | head -c 200)"
fi

echo ""
echo -e "  ${GREEN}$PASS passed${NC}, $([ "$FAIL" -gt 0 ] && echo -e "${RED}$FAIL failed${NC}" || echo "0 failed")"
[ "$FAIL" -eq 0 ]
