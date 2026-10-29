#!/bin/bash
# test_multi_model_load_failure.sh — two failure shapes under one temp --model-dir. weightless-model (a discoverable arch
# with no weight files) registers, then its first request is a 500 `model_load_failed`, state "error", and the error
# NAME crosses the inference-thread boundary intact (loadErrorFromName). unknown-arch-model (a model_type nothing
# serves) is refused by discovery and absent from /v1/models; an unknown id falls back to the default model by design
# (clients send ids like gpt-4o). A valid sibling keeps answering throughout.
#
#   ./tests/test_multi_model_load_failure.sh [valid_model] [port]

set -e
source "$(dirname "$0")/lib/server.sh"

VALID="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-8098}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$VALID" ]; then
    # Deliberately NOT a skip. This script skipped silently for months once its
    # default path moved to the external drive, and a skipped arm reads as a pass.
    echo -e "${RED}FAIL${NC} test_multi_model_load_failure: $VALID not found."
    echo "       pass a valid model dir as \$1 (the default path is stale)."
    exit 1
fi

TMPROOT=$(mktemp -d)
VALID_ID=$(basename "$VALID")
ln -s "$VALID" "$TMPROOT/$VALID_ID"
# Discoverable, loadable-looking, but shipping no weights.
mkdir -p "$TMPROOT/weightless-model"
cat > "$TMPROOT/weightless-model/config.json" <<EOF
{ "model_type": "qwen3", "hidden_size": 512, "num_hidden_layers": 2,
  "num_attention_heads": 8, "num_key_value_heads": 2, "intermediate_size": 1024,
  "vocab_size": 1000, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
  "rope_theta": 10000.0, "tie_word_embeddings": true }
EOF
cat > "$TMPROOT/weightless-model/tokenizer_config.json" <<EOF
{ "model_type": "qwen3" }
EOF

# Arch nothing serves — discovery must refuse this one outright.
mkdir -p "$TMPROOT/unknown-arch-model"
cat > "$TMPROOT/unknown-arch-model/config.json" <<EOF
{ "model_type": "not-a-real-model", "hidden_size": 1, "num_hidden_layers": 1 }
EOF

cleanup_root() { rm -rf "$TMPROOT"; }

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
"$BINARY" --model-dir "$TMPROOT" --model "$TMPROOT/$VALID_ID" --serve --port "$PORT" \
    ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
    cleanup_root
}
trap cleanup EXIT

srv_wait_health "$BASE" 30 1 || true

FAIL=0

echo "== weightless model returns a NAMED 500 =="
BODY=$(curl -s -o /dev/stdout -w '\n%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"weightless-model","messages":[{"role":"user","content":"Hi."}],"max_tokens":4}')
HTTP_STATUS=$(echo "$BODY" | tail -1)
ERR_BODY=$(echo "$BODY" | sed '$d')
if [ "$HTTP_STATUS" = "500" ]; then
    echo -e "${GREEN}PASS${NC} weightless-model → HTTP 500"
else
    echo -e "${RED}FAIL${NC} expected 500, got $HTTP_STATUS"
    FAIL=1
fi
# The error NAME is the point: a generic 500 means error_name was dropped
# crossing the inference-thread boundary (the class behind #144).
if echo "$ERR_BODY" | grep -q 'model_load_failed' && echo "$ERR_BODY" | grep -q 'Model load failed: [A-Za-z]'; then
    echo -e "${GREEN}PASS${NC} error names the load failure: $(echo "$ERR_BODY" | grep -oE 'Model load failed: [A-Za-z]+')"
else
    echo -e "${RED}FAIL${NC} 500 body carries no named load error: $ERR_BODY"
    FAIL=1
fi

echo
echo "== /v1/models reports weightless-model state=error =="
MODELS_JSON=$(curl -fs "$BASE/v1/models")
STATE=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(next((e.get('state','?') for e in d if e['id']=='weightless-model'), None))")
if [ "$STATE" = "error" ]; then
    echo -e "${GREEN}PASS${NC} weightless-model: state=error"
else
    echo -e "${RED}FAIL${NC} weightless-model state=$STATE (expected error)"
    FAIL=1
fi

echo
echo "== unrecognised arch is SKIPPED by discovery, not registered =="
UNKNOWN=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(next((e['id'] for e in d if e['id']=='unknown-arch-model'), 'absent'))")
if [ "$UNKNOWN" = "absent" ]; then
    echo -e "${GREEN}PASS${NC} unknown-arch-model absent from /v1/models"
else
    echo -e "${RED}FAIL${NC} unknown-arch-model registered as '$UNKNOWN' (discovery should refuse it)"
    FAIL=1
fi

echo
echo "== valid model still works =="
HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$VALID_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}")
if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}PASS${NC} valid model isolated from broken one"
else
    echo -e "${RED}FAIL${NC} valid model status=$HTTP_STATUS"
    FAIL=1
fi

exit $FAIL
