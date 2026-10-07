#!/bin/bash
# llama.cpp engine: concurrent requests decode together (#547), and a request
# decoding alone drafts with the model's MTP head (#548).
#
# Asserts, on one server with --llama-cache-entries 4:
#   1. Four concurrent chats all answer, and the log shows the batched step
#      engaged with more than one sequence.
#   2. The context armed MTP (`/props` settings.mtp.loaded) and a solo chat
#      ran MTP rounds that accepted drafts (`[spec-stats] mode=llama-mtp`).
#   3. With --llama-mtp-drafts 0 the same greedy chat answers the same text.
#
# Env:
#   LLAMA_GGUF_MODEL  A .gguf with an MTP head, e.g. unsloth/Qwen3.5-0.8B-MTP-GGUF
#                     Qwen3.5-0.8B-Q8_0.gguf. Unset = skip.
#   PORT              Default 19108.
#   BINARY            Default ./zig-out/bin/mlx-serve.
set -uo pipefail

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
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -30 "$LOG"; exit 1; }
        sleep 0.5
    done
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
