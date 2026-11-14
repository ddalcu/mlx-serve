#!/usr/bin/env bash
# Model Settings follow a model reached through a symlinked folder (issue #1184): the app keys
# `model-settings.json` by the folder's REAL path, discovery registers the link. Every reader
# must still find the entry: the unloaded row's context and alias, the per-request generation
# rules, and the load-time ctx/kv override.
#
# Runs under a private HOME. NEEDS A REAL MODEL: skips when MODEL is absent.
#
# Usage: MODEL=<model dir> ./tests/test_model_settings_symlink.sh [port]
source "$(dirname "$0")/_lib_models.sh"
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11386}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODEL="${MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit || echo "$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit")}"
MODEL="${MODEL%/}"
[ -f "$MODEL/config.json" ] || { echo "SKIP: needs a local chat model (MODEL=$MODEL)"; exit 0; }
REAL="$(cd "$MODEL" && pwd -P)"
ID="t/linked"
ALIAS="linky"

PASS=0; FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $1"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.mlx-serve"
LOG="$FAKE_HOME/server.log"
MODELS_ROOT="$FAKE_HOME/models"
mkdir -p "$MODELS_ROOT/t"
ln -s "$MODEL" "$MODELS_ROOT/$ID"
SRV=""
cleanup() {
    srv_stop "$SRV"
    rm -rf "$FAKE_HOME"
}
trap cleanup EXIT

# Keyed by the real folder, as the app writes it; the server only knows the link.
cat >"$FAKE_HOME/.mlx-serve/model-settings.json" <<JSON
{ "$REAL": { "ctx_size": 4096, "kv_quant": "8", "alias": "$ALIAS",
  "generation_defaults": { "max_tokens": { "value": 3, "ignore_client": true },
                           "enable_thinking": { "value": false, "ignore_client": true } } } }
JSON

HOME="$FAKE_HOME" "$BIN" --serve --model-dir "$MODELS_ROOT" --ctx-size 16384 --port "$PORT" \
    --log-level debug --log-file off >"$LOG" 2>&1 &
SRV=$!
UP=0
srv_wait_health "http://127.0.0.1:$PORT" 120 0.5 "$SRV" && UP=1
[ "$UP" = "1" ] || { echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1; }
URL="http://127.0.0.1:$PORT"

row() { curl -s "$URL/v1/models" | python3 -c "import json,sys; r=[m for m in json.load(sys.stdin)['data'] if m['id']==sys.argv[1]]; print(json.dumps(r[0]) if r else '')" "$1"; }
field() { python3 -c "import json,sys; d=json.loads(sys.argv[1] or 'null') or {}; v=d.get(sys.argv[2], (d.get('meta') or {}).get(sys.argv[2], '')); print(v)" "$1" "$2"; }

echo "[1] the unloaded row reads the real-path entry"
R="$(row "$ID")"
check "the link is registered as $ID" "$([ -n "$R" ] && echo 1 || echo 0)"
check "context_length is the pinned 4096 (got $(field "$R" context_length))" "$([ "$(field "$R" context_length)" = "4096" ] && echo 1 || echo 0)"
check "the row carries the alias (got $(field "$R" alias))" "$([ "$(field "$R" alias)" = "$ALIAS" ] && echo 1 || echo 0)"

echo "[2] a request by id applies the locked generation rules"
BODY=$(curl -s --max-time 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to twenty.\"}],\"max_tokens\":32}")
GOT=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',''))" 2>/dev/null)
check "locked max_tokens 3 outranks the client's 32 (got $GOT)" "$([ "$GOT" = "3" ] && echo 1 || echo 0)"

echo "[3] the load applied the ctx/kv override"
check "log names the override" "$(grep -q "\[model-settings\] .*ctx=4096 kv=8" "$LOG" && echo 1 || echo 0)"
R="$(row "$ID")"
check "loaded row context_length 4096 (got $(field "$R" context_length))" "$([ "$(field "$R" context_length)" = "4096" ] && echo 1 || echo 0)"
check "loaded row kv_quant 8 (got $(field "$R" kv_quant))" "$([ "$(field "$R" kv_quant)" = "8" ] && echo 1 || echo 0)"

echo "[4] the alias reaches the linked model"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/api/show" -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS\"}")
check "/api/show by alias -> 200 ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" = "0" ]
