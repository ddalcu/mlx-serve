#!/usr/bin/env bash
# Model Settings `alias` (issue #520): a short name for a model, accepted
# wherever a request names a model (chat, Ollama, load, unload), listed on its
# /v1/models row, and resolved by the keyless LAN gate like dispatch.
#
# The model is served twice from a private root: `t/twin` is the boot default
# and the only LAN-shared model, `t/main` carries the alias. An unknown name
# falls back to twin; only the alias reaches main, and a keyless LAN client
# may not use the alias to reach unshared main. Runs under a private HOME.
# NEEDS A REAL MODEL: skips when MODEL is absent.
#
# Usage: MODEL=<org/name dir> ./tests/test_model_alias.sh [port]
set -uo pipefail
PORT="${1:-11385}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODEL="${MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
MODEL="${MODEL%/}"
[ -f "$MODEL/config.json" ] || { echo "SKIP: needs a local chat model (MODEL=$MODEL)"; exit 0; }
ALIAS="shorty"

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
ln -s "$MODEL" "$MODELS_ROOT/t/main"
ln -s "$MODEL" "$MODELS_ROOT/t/twin"
ID="t/main"
SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    rm -rf "$FAKE_HOME"
}
trap cleanup EXIT
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 0.5

cat >"$FAKE_HOME/.mlx-serve/model-settings.json" <<JSON
{ "$MODELS_ROOT/t/main/": { "alias": "$ALIAS" }, "/nowhere/else": { "alias": "ghost" } }
JSON

HOME="$FAKE_HOME" "$BIN" --serve --model "$MODELS_ROOT/t/twin" --model-dir "$MODELS_ROOT" --port "$PORT" --lan-share t/twin --lan-name alias-test \
    --log-level debug --log-file off >"$LOG" 2>&1 &
SRV=$!
UP=0
for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
    sleep 0.5
done
[ "$UP" = "1" ] || { echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1; }
URL="http://127.0.0.1:$PORT"

row() { curl -s "$URL/v1/models" | python3 -c "import json,sys; r=[m for m in json.load(sys.stdin)['data'] if m['id']==sys.argv[1]]; print(json.dumps(r[0]) if r else '')" "$1"; }
field() { python3 -c "import json,sys; d=json.loads(sys.argv[1] or 'null'); print((d or {}).get(sys.argv[2], ''))" "$1" "$2"; }

echo "[1] /v1/models: the row keeps its id and carries the alias"
R="$(row "$ID")"
check "alias on the row" "$([ "$(field "$R" alias)" = "$ALIAS" ] && echo 1 || echo 0)"
check "no row is listed under the alias" "$([ -z "$(row "$ALIAS")" ] && echo 1 || echo 0)"

echo "[2] keyless LAN gate resolves the alias like dispatch"
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
if [ -z "$LAN_IP" ]; then
    echo "  SKIP: no en0 address"
else
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://$LAN_IP:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
    check "unknown name → the shared default passes ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://$LAN_IP:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}")
    check "alias of an UNSHARED model → 403, not the default's pass ($CODE)" "$([ "$CODE" = "403" ] && echo 1 || echo 0)"
    check "the unshared model was never loaded" "$([ "$(field "$(row "$ID")" loaded)" = "False" ] && echo 1 || echo 0)"
fi
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"ghost","messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
check "alias of an unregistered path falls back like any unknown name ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "... and still did not load the aliased model" "$([ "$(field "$(row "$ID")" loaded)" = "False" ] && echo 1 || echo 0)"

echo "[3] /v1/load-model + /v1/unload-model by alias"
BODY=$(curl -s "$URL/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS\"}")
check "load by alias answers with the canonical id" "$(echo "$BODY" | grep -q "\"id\":\"$ID\"" && echo 1 || echo 0)"
check "the model is now loaded" "$([ "$(field "$(row "$ID")" loaded)" = "True" ] && echo 1 || echo 0)"
BODY=$(curl -s "$URL/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS\"}")
check "unload by alias names the canonical id" "$(echo "$BODY" | grep -q "\"id\":\"$ID\"" && echo 1 || echo 0)"
check "the model is now unloaded" "$([ "$(field "$(row "$ID")" loaded)" = "False" ] && echo 1 || echo 0)"

echo "[4] chat by alias cold-loads and answers"
CODE=$(curl -s -o "$FAKE_HOME/chat.json" -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8}")
check "chat by alias → 200 ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "the aliased model served it" "$([ "$(field "$(row "$ID")" loaded)" = "True" ] && echo 1 || echo 0)"

echo "[5] Ollama: a tagged alias resolves"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/api/show" -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS:latest\"}")
check "/api/show with $ALIAS:latest → 200 ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/api/chat" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false,\"options\":{\"num_predict\":4}}")
check "/api/chat with $ALIAS:latest → 200 ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"

echo "[6] an alias edit on a running server takes effect without a restart"
cat >"$FAKE_HOME/.mlx-serve/model-settings.json" <<JSON
{ "$MODELS_ROOT/t/main/": { "alias": "renamed" } }
JSON
sleep 1.5
check "the row carries the new alias" "$([ "$(field "$(row "$ID")" alias)" = "renamed" ] && echo 1 || echo 0)"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/api/show" -H 'Content-Type: application/json' -d '{"model":"renamed"}')
check "the new alias resolves ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/api/show" -H 'Content-Type: application/json' -d "{\"model\":\"$ALIAS\"}")
check "the old alias no longer does ($CODE)" "$([ "$CODE" = "404" ] && echo 1 || echo 0)"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" = "0" ]
