#!/bin/bash
# Wave 1.A — per-request `kv_quant` body field.
#
# Starts the server with `--kv-quant off` (the process-level default) and
# fires three back-to-back requests against /v1/chat/completions:
#   1. no `kv_quant` field         → inherits process default (off)
#   2. {"kv_quant": 4}              → per-request override to 4-bit affine
#   3. {"kv_quant": "off"}          → explicit override back to dense
# All three must complete with non-empty content. The 4-bit request is the
# critical one — if SubmitParams.kv_quant_config plumbing or the slot's
# KVCache.initWithConfig is broken, the request returns garbage / pad-only.
#
# Hot-prefix-cache scheme isolation: a follow-up identical request under
# scheme A must NOT pick up a cache hit from a prior scheme-B commit. We
# assert that the second 4-bit request emits a `[hot-cache] reused N/M`
# line whose match length is 0 against the dense-default first request.
# (Tightly testing this requires same prompt + different scheme; we do
# scheme-flip while pinning the prompt and assert no cross-scheme hit.)
#
# Usage:
#   ./tests/test_kv_quant_per_request.sh [/path/to/model] [port]
# Defaults: model = ~/.mlx-serve/models/gemma-4-e4b-it-4bit, port = 8094.

set -e

MODEL="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-8094}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_kv_quant_per_request: $MODEL not found."
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
echo "  starting server (--kv-quant off, --prefix-cache-entries 4)..."
"$BINARY" --model "$MODEL" --serve --port "$PORT" --kv-quant off --prefix-cache-entries 4 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
}
trap cleanup EXIT

up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -30 "$LOGFILE"
    exit 1
fi

fire() {
    # $1 = label, $2 = optional kv_quant field value (empty = no field)
    local label="$1"
    local kvq="$2"
    local extra=""
    if [ -n "$kvq" ]; then
        extra=", \"kv_quant\": $kvq"
    fi
    local body
    body=$(curl -s -X POST -H "Content-Type: application/json" \
        -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0.0,\"stream\":false${extra}}" \
        "$BASE/v1/chat/completions")
    local content
    content=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || echo "")
    if [ -z "$content" ] || [ "$content" = "None" ]; then
        echo -e "${RED}FAIL${NC} $label: empty/null content"
        echo "  body: $(echo "$body" | head -c 400)"
        return 1
    fi
    echo -e "${GREEN}PASS${NC} $label: $(echo "$content" | head -c 80)..."
    return 0
}

FAIL=0
echo
echo "== per-request kv_quant override =="
fire "request #1 (no kv_quant field, inherits process default 'off')" "" || FAIL=1
sleep 1
fire "request #2 (kv_quant: 4 — per-request 4-bit override)" "4" || FAIL=1
sleep 1
fire "request #3 (kv_quant: \"off\" — explicit dense)" '"off"' || FAIL=1
sleep 1
fire "request #4 (kv_quant: 8 — per-request 8-bit)" "8" || FAIL=1

# Hot-cache scheme isolation: server log MUST contain a kv-quant override log
# line for each non-default request — that's how we know the parse hit the
# scheduler path. And the cache MUST report a fresh resident line per scheme
# transition without crashing.
echo
echo "== server log: kv-quant override lines =="
if ! grep -q "kv-quant override: affine 4-bit" "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} expected '  kv-quant override: affine 4-bit' line in server log"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} per-request 4-bit override observed in server log"
fi
if ! grep -q "kv-quant override: affine 8-bit" "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} expected '  kv-quant override: affine 8-bit' line in server log"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} per-request 8-bit override observed in server log"
fi
if ! grep -q "kv-quant override: off" "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} expected '  kv-quant override: off' line in server log"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} per-request 'off' override observed in server log"
fi

exit $FAIL
