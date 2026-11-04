#!/bin/bash
# test_llama_concurrent.sh — llama.cpp sequences + MTP on one server (--llama-cache-entries 4): four concurrent chats
# answer with `[batched] llama.cpp decode engaged (seqs=N)`; `/props` settings.mtp.loaded on a GGUF with a head; a
# solo greedy chat accepts drafts (`[spec-stats] mode=llama-mtp`); --llama-mtp-drafts 0 answers the same text.
#
#   LLAMA_GGUF_MODEL=<Qwen3.5-0.8B-MTP-GGUF Q8_0 file> ./tests/test_llama_concurrent.sh   # PORT (19108), BINARY
#
# Unset = SKIP. Hermetic twins: `pickLlamaSeq`, `llamaDraftsThisTick`, `findLlamaMtpSidecar`; model-gated: the
# LLAMA_TEST_MODEL tests in src/engines/llama/bridge.zig.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

MODEL="${LLAMA_GGUF_MODEL:-}"
PORT="${PORT:-19108}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -n "$MODEL" ] || { echo "SKIP test_llama_concurrent: set LLAMA_GGUF_MODEL=/path/to/model.gguf"; exit 0; }
[ -f "$MODEL" ] || { echo "SKIP: GGUF file missing: $MODEL"; exit 0; }
[ -x "$BIN" ] || { echo "fail: build mlx-serve first ($BIN)"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1"; }

LOG="$(mktemp)"
OUT="$(mktemp -d)"
SERVER_PID=""
stop() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}
trap 'stop; rm -rf "$LOG" "$OUT"' EXIT INT TERM

start() {
    "$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 4096 --log-level info "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 240 0.5 "$SERVER_PID" && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -30 "$LOG"; exit 1; }
    echo "fail: server never became healthy"; exit 1
}

ask() {
    curl -sf --max-time 180 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$1" '{messages:[{role:"user",content:$p}],max_tokens:64,temperature:0,stream:false}')"
}

SOLO="Count from one to twenty in words, separated by commas."

echo "==> server with 4 sequences"
start --llama-cache-entries 4

echo "==> four concurrent chats"
i=0
pids=()
for p in "Name three primary colors." "Write a haiku about rain." "What is 17 times 23?" "List the planets of the solar system."; do
    ask "$p" > "$OUT/$i.json" &
    pids+=($!)
    i=$((i+1))
done
wait "${pids[@]}" # never a bare wait: the server is a background job too
answered=0
for f in "$OUT"/*.json; do
    [ -n "$(jq -r '.choices[0].message.content // empty' "$f" 2>/dev/null)" ] && answered=$((answered+1))
done
[ "$answered" = 4 ] && ok "all four concurrent chats answered" || bad "only $answered/4 concurrent chats answered"
grep -q "\[batched\] llama.cpp decode engaged (seqs=" "$LOG" && ok "batched step engaged" || bad "no batched llama.cpp step in the log"

echo "==> solo chat with the MTP head"
mtp_loaded=$(curl -sf "$BASE/props" | jq -r '.settings.mtp.loaded // empty')
if [ "$mtp_loaded" != "true" ]; then
    bad "settings.mtp.loaded=$mtp_loaded — does $MODEL ship an MTP head?"
else
    ok "MTP armed"
    WITH_MTP="$(ask "$SOLO" | jq -r '.choices[0].message.content')"
    accepts=$(grep -o "mode=llama-mtp attempts=[0-9]* accepts=[0-9]*" "$LOG" | tail -1 | sed 's/.*accepts=//')
    [ "${accepts:-0}" -gt 0 ] && ok "MTP rounds accepted $accepts drafts" || bad "no accepted MTP drafts (accepts=${accepts:-none})"

    stop
    echo "==> same chat with --llama-mtp-drafts 0"
    start --llama-cache-entries 4 --llama-mtp-drafts 0
    PLAIN="$(ask "$SOLO" | jq -r '.choices[0].message.content')"
    grep -q "mode=llama-mtp" "$LOG" && bad "MTP ran with --llama-mtp-drafts 0" || ok "no MTP rounds when off"
    [ "$WITH_MTP" = "$PLAIN" ] && ok "MTP output equals plain greedy output" || bad "MTP and plain greedy differ: '$WITH_MTP' vs '$PLAIN'"
fi

echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = 0 ]
