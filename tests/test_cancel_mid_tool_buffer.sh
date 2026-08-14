#!/bin/bash
# Guard: disconnecting WHILE the stream is buffering for tool detection leaves
# no leak and no wedge.
#
# With tools present the server emits nothing until the pattern resolves — it
# holds tokens, sometimes for a long one-shot write_file. That buffering window
# is the one place a client can vanish while the server owns an unflushed
# buffer AND a live slot, and it is not covered by test_disconnect_cancel.sh
# (which cancels a plain prose stream, where every token has already gone out).
#
# The test drops the connection mid-buffer, several times, then asserts:
#   • the server cancels the slot rather than generating to max_tokens
#   • memory does not climb across the cycles (the buffer is freed)
#   • the server still answers, with tools and without, afterwards
#
# Usage: CANCEL_TEST_MODEL=<dir> ./tests/test_cancel_mid_tool_buffer.sh [port]
#   CANCEL_ROUNDS=5

set -u

MODEL="${CANCEL_TEST_MODEL:-}"
PORT="${1:-8137}"
ROUNDS="${CANCEL_ROUNDS:-5}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; DIM='\033[2m'; NC='\033[0m'
PASS=0; FAIL=0
ok()  { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
bad() { echo -e "  ${RED}FAIL${NC} $1"; shift; for l in "$@"; do echo "        $l"; done; FAIL=$((FAIL+1)); }

[ -n "$MODEL" ] || { echo "SKIP: CANCEL_TEST_MODEL not set"; exit 0; }
[ -f "$MODEL/config.json" ] || { echo "SKIP: no config.json at $MODEL"; exit 0; }
[ -x ./zig-out/bin/mlx-serve ] || { echo "FAIL: build first"; exit 1; }

LOG=$(mktemp /tmp/cancel_toolbuf.XXXXXX)
pkill -f "bin/mlx-serve" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve --model "$MODEL" --serve --port "$PORT" --log-level debug > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT
for i in $(seq 1 300); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not start"; exit 1; }

mem_bytes() { curl -sf -m 10 "$BASE/props" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('memory') or {}).get('active_bytes') or 0)" 2>/dev/null || echo 0; }

# A request whose answer is ONE big tool call: the server buffers the whole
# thing for detection, so a disconnect lands mid-buffer with high probability.
REQ=$(python3 <<'PY'
import json
print(json.dumps({"model": "mlx-serve",
 "messages": [{"role": "user", "content":
   "Use write_file RIGHT NOW to create a complete standalone HTML page saved as "
   "mars.html about Mars: full <!DOCTYPE html>, a <head> with a <title>, an "
   "embedded <style> block, an <h1>, and at least eight <p> paragraphs. Call the "
   "tool now; do not ask questions."}],
 "tools": [{"type": "function", "function": {
   "name": "write_file", "description": "Write a file",
   "parameters": {"type": "object", "properties": {
     "path": {"type": "string"}, "content": {"type": "string"}},
     "required": ["path", "content"]}}}],
 "max_tokens": 3000, "temperature": 0.7, "stream": True}))
PY
)

MEM_BEFORE=$(mem_bytes)
CANCELS_BEFORE=$(grep -c "\[client_disconnect\]" "$LOG" 2>/dev/null); CANCELS_BEFORE="${CANCELS_BEFORE:-0}"

for r in $(seq 1 "$ROUNDS"); do
    # --max-time drops the socket mid-generation; the buffer is still unflushed
    # because nothing is emitted until the tool pattern resolves.
    curl -sN -m 3 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary "$REQ" >/dev/null 2>&1
    echo -e "  ${DIM}round $r: disconnected after 3s${NC}"
    sleep 2
done

# Give the server a moment to notice the last drop.
sleep 5

CANCELS_AFTER=$(grep -c "\[client_disconnect\]" "$LOG" 2>/dev/null); CANCELS_AFTER="${CANCELS_AFTER:-0}"
NEW_CANCELS=$((CANCELS_AFTER - CANCELS_BEFORE))
if [ "$NEW_CANCELS" -ge "$ROUNDS" ]; then
    ok "every disconnect ended its request as [client_disconnect] ($NEW_CANCELS for $ROUNDS rounds)"
else
    bad "disconnects cancel the slot" "$NEW_CANCELS [client_disconnect] endings for $ROUNDS rounds — a slot generated on after the client left"
fi

# Stronger than the log line: the slot must have STOPPED, not run to max_tokens.
# 3 s of generation is a few hundred tokens; max_tokens is 3000.
OVERRUN=$(grep -aoE "<- [0-9]+\+([0-9]+) tokens streamed" "$LOG" 2>/dev/null \
    | sed -E 's/.*\+([0-9]+) tokens.*/\1/' | awk '$1 > 2000' | wc -l | tr -d ' ')
if [ "${OVERRUN:-0}" -eq 0 ]; then
    ok "no cancelled request generated on toward max_tokens"
    grep -aoE "<- [0-9]+\+[0-9]+ tokens streamed.*\[client_disconnect\]" "$LOG" 2>/dev/null | tail -2 | sed 's/^/        /'
else
    bad "a cancelled request kept generating" "$OVERRUN request(s) produced >2000 of 3000 tokens after the client left"
fi

# A cancelled buffer must be freed, not retained per round.
sleep 3
MEM_AFTER=$(mem_bytes)
if [ "${MEM_BEFORE:-0}" -gt 0 ] && [ "${MEM_AFTER:-0}" -gt 0 ]; then
    GROWTH=$(( (MEM_AFTER - MEM_BEFORE) * 100 / MEM_BEFORE ))
    if [ "$GROWTH" -lt 25 ]; then
        ok "no runaway retention across $ROUNDS cancelled buffers (active ${GROWTH}%)"
    else
        bad "cancelled tool buffers are retained" "active grew ${GROWTH}% over $ROUNDS rounds ($MEM_BEFORE -> $MEM_AFTER)"
    fi
else
    echo -e "  ${DIM}note: /props memory unavailable — retention not measured${NC}"
fi

# Not wedged: the next request must work, with tools and without.
R=$(curl -sf -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' 2>/dev/null \
    | python3 -c "import json,sys; print((json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip())" 2>/dev/null)
[ -n "$R" ] && ok "plain request works after the cancels (${R:0:30})" || bad "server wedged: plain request returned nothing"

R2=$(echo "$REQ" | python3 -c "
import json,sys
b=json.load(sys.stdin); b['stream']=False; b['max_tokens']=400; print(json.dumps(b))" \
    | curl -sf -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @- 2>/dev/null \
    | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
tcs=m.get('tool_calls') or []
if not tcs: print('nocall'); raise SystemExit
try: json.loads(tcs[0]['function']['arguments']); print('ok')
except Exception: print('badjson')" 2>/dev/null)
case "$R2" in
    ok)      ok "tool-calling still works after the cancels" ;;
    nocall)  bad "tool call missing after the cancels" ;;
    badjson) bad "tool arguments invalid after the cancels" ;;
    *)       bad "tools request failed after the cancels" "$R2" ;;
esac

grep -qiE "panic|segmentation fault|SIGBUS|unreachable" "$LOG" \
    && bad "crash signature in the server log" "$(grep -iE 'panic|segmentation|SIGBUS|unreachable' "$LOG" | head -2)" \
    || ok "no crash signature in the server log"

echo
echo "cancel mid-tool-buffer: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
