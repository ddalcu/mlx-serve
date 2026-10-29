#!/bin/bash
# test_lan_peer_key.sh — a LAN peer running --api-key: A shares with `--api-key K --api-key-strict` (strict, since
# same-machine discovery arrives over loopback, which a plain key exempts); B discovers with a scratch HOME.
#
#   ./tests/test_lan_peer_key.sh [/path/to/model] [portA]
#
# Without a key B lists A under `lan_needs_key` with no models; writing B's ~/.mlx-serve/lan-keys.json brings A's
# models in within seconds; a chat on `<id>@<A>` completes through B. SKIPs without the model or a non-loopback IP.
# Hermetic twins: `peerKeyFromJson`, the keyed-peer fetch and tunnel `Authorization` line (host/lan/lan.zig), LanPeerKeysTests.

source "$(dirname "$0")/lib/server.sh"
source "$(dirname "$0")/_lib_models.sh"
MODEL="${1:-$(find_model mlx-community/gemma-4-e4b-it-4bit mlx-community/gemma-4-e4b-it-8bit)}"
PORT_A="${2:-11321}"
PORT_B=$((PORT_A + 1))
PEER_NAME="keytest-a-$$"
KEY="peer-key-$$"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_lan_peer_key: no gemma-4-e4b checkpoint found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi
if [ -z "$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)" ]; then
    echo -e "${YELLOW}SKIP${NC} test_lan_peer_key: no non-loopback IP (offline Mac)."
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT_A" 2>/dev/null || true
pkill -f "mlx-serve.*--port $PORT_B" 2>/dev/null || true
sleep 1

EMPTY_DIR=$(mktemp -d)
HOME_B=$(mktemp -d)
LOG_A=$(mktemp); LOG_B=$(mktemp)
PID_A=""; PID_B=""
cleanup() {
    [ -n "$PID_A" ] && kill "$PID_A" 2>/dev/null || true
    [ -n "$PID_B" ] && kill "$PID_B" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$EMPTY_DIR" "$HOME_B" "$LOG_A" "$LOG_B"
}
trap cleanup EXIT

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo -e "${GREEN}PASS${NC} $1"; }
bad()  { FAIL=$((FAIL+1)); echo -e "${RED}FAIL${NC} $1"; }
check() { if [ "$2" -eq 0 ]; then ok "$1"; else bad "$1"; fi; }

wait_health() { # <port>
    srv_wait_health "http://127.0.0.1:$1" 120 1 && return 0
    return 1
}

# Prints "<needs-key 0|1> <first remote id or ->" for B's /v1/models.
peer_state() {
    curl -s "http://127.0.0.1:$PORT_B/v1/models" | python3 -c '
import json,sys
peer = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    print("0 -"); sys.exit()
ids = [m["id"] for m in d.get("data", []) if m.get("id","").endswith("@" + peer)]
print(("1" if peer in d.get("lan_needs_key", []) else "0"), (ids[0] if ids else "-"))' "$PEER_NAME"
}

echo "── Booting A (keyed, shares '$(basename "$MODEL")' as \"$PEER_NAME\") + B (headless, discovering)"
"$BINARY" --model "$MODEL" --serve --port "$PORT_A" --log-level debug --api-key "$KEY" --api-key-strict \
    --lan-share all --lan-name "$PEER_NAME" --log-file off >"$LOG_A" 2>&1 &
PID_A=$!
HOME="$HOME_B" "$BINARY" --serve --port "$PORT_B" --model-dir "$EMPTY_DIR" --log-level debug \
    --lan-discover --log-file off >"$LOG_B" 2>&1 &
PID_B=$!
wait_health "$PORT_A" || { bad "server A never became healthy"; tail -20 "$LOG_A"; exit 1; }
wait_health "$PORT_B" || { bad "server B never became healthy"; tail -20 "$LOG_B"; exit 1; }
ok "both servers healthy"

# ── 1. No key: listed as needing one, no models ──
STATE="0 -"
for _ in $(seq 1 45); do
    STATE=$(peer_state)
    [ "${STATE%% *}" = "1" ] && break
    sleep 1
done
[ "$STATE" = "1 -" ]; check "keyed peer is listed under lan_needs_key with no models (got: $STATE)" $?

# ── 2. Saving the key brings the models in ──
mkdir -p "$HOME_B/.mlx-serve"
printf '{"%s": "%s"}\n' "$PEER_NAME" "$KEY" > "$HOME_B/.mlx-serve/lan-keys.json"
for _ in $(seq 1 15); do
    STATE=$(peer_state)
    [ "${STATE#* }" != "-" ] && break
    sleep 1
done
RID="${STATE#* }"
[ "${STATE%% *}" = "0" ] && [ "$RID" != "-" ]; check "saved key unlocks the peer's models (got: $STATE)" $?

# ── 3. A chat through B carries the key ──
if [ "$RID" != "-" ]; then
    curl -s -m 120 "http://127.0.0.1:$PORT_B/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$RID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],\"max_tokens\":30,\"temperature\":0}" |
        python3 -c 'import json,sys; c = json.load(sys.stdin)["choices"][0]["message"]["content"]; assert c and c.strip()' 2>/dev/null
    check "chat on the keyed peer's model completes through B" $?
fi

echo ""
echo "── test_lan_peer_key: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
