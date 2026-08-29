#!/bin/bash
# Regression tests for client-disconnect handling during long prefills.
#
# Pins the 2026-06-10 live failure: Claude Code with a huge MCP toolset sent a
# ~40K-token prompt to gemma-4-12b, timed out waiting on the multi-minute cold
# prefill, disconnected, and RETRIED — but the server only checks the peer
# socket after a token arrives, so the handler sat blocked in waitNext for the
# whole prefill. Abandoned prefills piled up serially; `lsof` showed ZERO
# clients connected while the GPU ground away for minutes. The server looked
# dead ("not returning anything") and memory climbed with each ghost KV.
#
# Two contracts:
#   1. Keepalive: during a long prefill, streaming responses emit periodic
#      SSE pings (Anthropic `event: ping` / OpenAI `: keepalive` comments) so
#      clients don't hit their idle timeout in the first place.
#   2. Disconnect-cancel: when the client vanishes mid-prefill, the server
#      notices within one keepalive interval, cancels the slot (aborting the
#      prefill at the next chunk boundary), and the next request isn't stuck
#      behind a ghost.
#
# Usage: ./tests/test_disconnect_cancel.sh [model_dir] [port]
#   Starts its own server. Default model: Gemma 4 E4B 8-bit.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
PORT="${2:-11264}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_disconnect_cancel.log
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
# --no-pld for predictable prefill timing; big ctx for the long prompt.
"$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size 32768 --no-pld --log-level debug > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; exit 1; }

# ~27K-token unique-ish prompts (≈40-60s cold prefill on E4B) — slow enough
# that keepalives and the disconnect window are clearly observable. Distinct
# seeds per check: a shared prompt would let check 2's ghost ride check 1's
# hot-prefix-cache entry and skip the cold prefill entirely.
big_body() { # $1 = seed -> JSON on stdout
    python3 - "$1" <<'EOF'
import json, random, sys
random.seed(int(sys.argv[1]))
words = ["alpha","bridge","cobalt","delta","ember","fjord","glacier","harbor",
         "isotope","jasper","kelvin","lumen","meridian","nectar","onyx","prism"]
text = " ".join(random.choice(words) + str(i % 97) for i in range(9000))
print(json.dumps({
    "model": "m", "max_tokens": 40, "stream": True,
    "messages": [{"role": "user", "content": "Summarize this in one word: " + text}],
}))
EOF
}
big_body 7 > /tmp/disconnect_cancel_big.json
big_body 1234 > /tmp/disconnect_cancel_big2.json

echo "1. keepalive pings flow during a long prefill (/v1/messages stream)"
curl -sN -m 120 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d @/tmp/disconnect_cancel_big.json > /tmp/disconnect_keepalive.sse
PINGS=$(grep -c '"type":"ping"' /tmp/disconnect_keepalive.sse)
echo "    -> $PINGS ping event(s)"
check "≥2 ping events (initial + ≥1 during prefill)" "$([ "$PINGS" -ge 2 ] && echo 1 || echo 0)"
grep -q '"type":"message_stop"' /tmp/disconnect_keepalive.sse
check "request still completes normally" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo "2. disconnect mid-prefill cancels the ghost; next request not blocked"
# Client gives up after 3s — mirrors Claude Code's timeout+retry behavior.
curl -sN -m 3 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d @/tmp/disconnect_cancel_big2.json > /dev/null 2>&1
START=$(date +%s)
SMALL=$(curl -s -m 60 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d '{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"Say OK."}]}')
ELAPSED=$(( $(date +%s) - START ))
echo "    -> follow-up request took ${ELAPSED}s"
echo "$SMALL" | grep -q '"type":"message"'
check "follow-up request succeeded" "$([ $? -eq 0 ] && echo 1 || echo 0)"
# Post-fix: ghost noticed within one keepalive interval (10s) + chunk abort.
# Pre-fix: the follow-up waits out the ghost's entire remaining prefill+decode.
check "follow-up completed in <12s (ghost cancelled)" "$([ "$ELAPSED" -lt 12 ] && echo 1 || echo 0)"
grep -q "client disconnected" "$LOG"
check "server logged the disconnect-cancel" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "===== $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ] || exit 1
