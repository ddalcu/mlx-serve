#!/usr/bin/env bash
# The sushi guest engine end to end: a guest this host cannot run, or --no-sushi,
# fails the load by name; a Qwen3.8-Flash-Next EXL3 pack is listed as engine
# "sushi", loads by starting the guest (the pinned release, downloaded on first
# use), serves chat / messages / responses through the byte-for-byte forward, and
# reports the guest's /props; the guest's own port refuses a request without its
# key; Ollama is refused by name; unload stops the guest; the guest and an MLX
# model swap (one engine at a time) and a model mid-request is never evicted for
# the guest; a guest that dies fails its entry naming its log until unloaded; the
# guest exits when mlx-serve is killed (--parent-pid).
#
# SKIPs without a pack: set SUSHI_PACK=<dir>, or keep one under
# ~/.mlx-serve/models/<org>/. Needs ~55 GB free for the 3bpw pack. The engine-swap
# checks also need a small MLX model: SUSHI_MLX_MODEL=<dir>. The embedding-encoder
# exception to the one-engine rule is covered by the model_registry unit tests.

set -euo pipefail

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11302}"
BASE="http://127.0.0.1:$PORT"

PACK="${SUSHI_PACK:-}"
if [[ -z "$PACK" ]]; then
    for c in "$HOME"/.mlx-serve/models/*/Qwen3.8-Flash-Next-Sushi-*; do
        [[ -f "$c/config.json" ]] && PACK="$c" && break
    done
fi
if [[ -z "$PACK" || ! -f "$PACK/config.json" ]]; then
    echo "[skip] no Sushi pack — set SUSHI_PACK=<dir>"
    exit 0
fi
if [[ ! -x "$BINARY" ]]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

WORK="$(mktemp -d)"
LOG="$WORK/mlx-serve.log"
mkdir -p "$WORK/models"
ln -s "$PACK" "$WORK/models/"
ID="$(basename "$PACK")"
MLX_ID=""
if [[ -n "${SUSHI_MLX_MODEL:-}" && -f "$SUSHI_MLX_MODEL/config.json" ]]; then
    ln -s "$SUSHI_MLX_MODEL" "$WORK/models/"
    MLX_ID="$(basename "$SUSHI_MLX_MODEL")"
fi
SERVER_PID=""
GUEST_PID=""
WS_PID=""

cleanup() {
    [[ -n "$WS_PID" ]] && kill "$WS_PID" 2>/dev/null || true
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
    [[ -n "$GUEST_PID" ]] && kill "$GUEST_PID" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
    echo "[fail] $*"
    tail -30 "$LOG" 2>/dev/null || true
    exit 1
}

json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

boot() {
    env HOME="${BOOT_HOME:-$HOME}" "$BINARY" serve --model-dir "$WORK/models" --host 127.0.0.1 --port "$PORT" --log-file "$LOG" "$@" >/dev/null 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 60); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited during boot"
        sleep 1
    done
    fail "/health never came up"
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

load_code() {
    curl -s -o "$WORK/load.json" -w '%{http_code}' --max-time "${2:-1800}" -X POST "$BASE/v1/load-model" \
        -H 'Content-Type: application/json' -d "{\"model\":\"${1:-$ID}\"}"
}

load() {
    local code
    code=$(load_code)
    [[ "$code" == "200" ]] || fail "load-model returned $code: $(head -c 300 "$WORK/load.json")"
    GUEST_PID="$(pgrep -P "$SERVER_PID" -x sushi || true)"
    [[ -n "$GUEST_PID" ]] || fail "no sushi child of mlx-serve pid $SERVER_PID"
}

row() { curl -sf "$BASE/v1/models" | json "[m for m in d['data'] if m['id']=='${2:-$ID}'][0]$1"; }

guest_gone() {
    for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
    return 1
}

FAKE_LOG="$WORK/home/.mlx-serve/logs/sushi-$PORT.log"
for api in 2 1; do
    printf '#!/bin/sh\n[ "$1" = --guest-manifest ] && { echo %s; exit 0; }\necho "fake guest: refusing to serve" >&2\nexit 3\n' \
        "'{\"guest_api\":$api}'" >"$WORK/fake-sushi"
    chmod +x "$WORK/fake-sushi"
    BOOT_HOME="$WORK/home" boot --sushi-path "$WORK/fake-sushi"
    CODE=$(load_code)
    if [[ "$api" == 2 ]]; then
        [[ "$CODE" == "500" ]] && grep -q "SushiGuestApiUnsupported" "$WORK/load.json" ||
            fail "a guest_api 2 guest loaded ($CODE): $(head -c 300 "$WORK/load.json")"
        echo "[ok] a guest speaking guest_api 2 is refused by name"
    else
        [[ "$CODE" == "500" ]] && grep -qF "SushiGuestExited (log: $FAKE_LOG)" "$WORK/load.json" &&
            grep -q "fake guest: refusing to serve" "$FAKE_LOG" ||
            fail "a guest that died on start answered $CODE: $(head -c 300 "$WORK/load.json")"
        [[ "$(row "['state']")" == "error" ]] || fail "a guest that died on start left its entry '$(row "['state']")'"
        echo "[ok] a guest that dies on start fails the load, naming its log"
    fi
    stop_server
done

BOOT_HOME="$WORK/home" boot --no-sushi
if [[ -n "$MLX_ID" ]]; then
    [[ "$(load_code "$MLX_ID")" == "200" ]] || fail "could not load $MLX_ID"
fi
CODE=$(load_code)
[[ "$CODE" == "500" ]] && grep -q "SushiDisabled" "$WORK/load.json" || fail "--no-sushi answered $CODE: $(head -c 300 "$WORK/load.json")"
[[ -z "$MLX_ID" || "$(row "['state']" "$MLX_ID")" == "ready" ]] || fail "--no-sushi evicted $MLX_ID for a load that could not start"
echo "[ok] --no-sushi refuses the pack by name${MLX_ID:+, $MLX_ID stays loaded}"
stop_server

boot
echo "[ok] mlx-serve up on $PORT (pack: $ID)"

[[ "$(row "['meta']['engine']")" == "sushi" ]] || fail "unloaded row does not name the sushi engine"
[[ "$(row "['state']")" == "unloaded" ]] || fail "pack loaded before it was asked for"
echo "[ok] /v1/models lists the pack as engine sushi, unloaded"

START=$(date +%s)
load
echo "[ok] loaded in $(( $(date +%s) - START ))s, guest pid $GUEST_PID"
grep -q "\[sushi\] guest ready" "$LOG" || fail "no '[sushi] guest ready' line"

CTX="$(row "['context_length']")"
[[ "$(row "['meta']['engine']")" == "sushi" && "$(row "['state']")" == "ready" && "$CTX" -gt 0 ]] ||
    fail "ready row: engine/state/context wrong (context_length=$CTX)"
echo "[ok] ready row: engine sushi, context_length $CTX"

ENGINE="$(curl -sf "$BASE/props?model=$ID" | json "d['settings']['engine']")"
[[ "$ENGINE" == "sushi" ]] || fail "/props settings.engine is '$ENGINE'"
echo "[ok] /props is the guest's, settings.engine sushi"

GUEST_SETTINGS="$(curl -sf "$BASE/props?model=$ID" | json "d['settings']['kv_quant'] + ' ' + str(d['settings']['mtp']['loaded'])")"
ROW_SETTINGS="$(curl -sf "$BASE/v1/models" | json "[m['meta']['kv_quant'] + ' ' + str(m['meta']['mtp_loaded']) for m in d['data'] if m['id']=='$ID'][0]")"
[[ "$ROW_SETTINGS" == "$GUEST_SETTINGS" ]] || fail "ready row reports kv/mtp '$ROW_SETTINGS', the guest runs '$GUEST_SETTINGS'"
GUEST_QUANT="$(grep "\[sushi\] guest ready" "$LOG" | tail -1 | sed 's/.*quantization "\(.*\)"$/\1/')"
ROW_QUANT="$(row "['meta']['quantization']")"
[[ -n "$GUEST_QUANT" && "$ROW_QUANT" == "$GUEST_QUANT" ]] || fail "ready row reports quantization '$ROW_QUANT', the guest reported '$GUEST_QUANT'"
echo "[ok] ready row reports the guest's own kv_quant, mtp_loaded ($ROW_SETTINGS) and quantization ($ROW_QUANT)"

GUEST_PORT=$(grep -o "guest pid=$GUEST_PID port=[0-9]*" "$LOG" | tail -1 | cut -d= -f3)
CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$GUEST_PORT/v1/models")
[[ "$CODE" == "401" ]] || fail "the guest's own port answered $CODE without its key"
echo "[ok] the guest's port :$GUEST_PORT refuses a request without its key"

CHAT="{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colors.\"}],\"max_tokens\":48,\"temperature\":0,\"enable_mtp\":false"
curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$CHAT}" >"$WORK/plain.json" ||
    fail "non-stream chat failed"
curl -sfN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$CHAT,\"stream\":true}" >"$WORK/stream.sse" ||
    fail "stream chat failed"
python3 - "$WORK/plain.json" "$WORK/stream.sse" <<'EOF' || fail "stream and non-stream bytes differ"
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
plain = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
reasoning, content, done = "", "", False
for line in open(sys.argv[2]):
    if line.startswith("data: [DONE]"):
        done = True
    elif line.startswith("data: "):
        for ch in json.loads(line[6:]).get("choices", []):
            d = ch.get("delta", {})
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
assert done, "no [DONE]"
assert plain.strip("\x00"), "empty answer"
if plain != reasoning + "\x00" + content:
    print("non-stream:", repr(plain)); print("stream:    ", repr(reasoning + "\x00" + content))
    sys.exit(1)
print("[ok] chat: stream and non-stream are the same bytes:", repr(plain.strip("\x00")[:80]))
EOF

MSG=$(curl -sf "$BASE/v1/messages" -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
    -d "{\"model\":\"$ID\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}") ||
    fail "/v1/messages failed"
[[ "$(printf '%s' "$MSG" | json "d['type'] + ':' + str(len(d['content']) > 0)")" == "message:True" ]] ||
    fail "/v1/messages answered: $(printf '%s' "$MSG" | head -c 300)"
echo "[ok] /v1/messages answers through the guest"

RESP=$(curl -sf "$BASE/v1/responses" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"input\":\"Say hi.\",\"max_output_tokens\":32}") || fail "/v1/responses failed"
[[ "$(printf '%s' "$RESP" | json "d['object']")" == "response" ]] || fail "/v1/responses answered: $(printf '%s' "$RESP" | head -c 300)"
echo "[ok] /v1/responses answers through the guest"

CODE=$(curl -s -o "$WORK/ollama.json" -w '%{http_code}' "$BASE/api/chat" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}")
[[ "$CODE" == "400" ]] && grep -q "sushi engine" "$WORK/ollama.json" || fail "Ollama /api/chat returned $CODE: $(head -c 200 "$WORK/ollama.json")"
echo "[ok] Ollama /api/chat refused by name"

N=$(grep -c "\[sushi\] POST /v1/" "$LOG" || true)
(( N >= 4 )) || fail "only $N forwarded requests in the log"
echo "[ok] $N requests engaged the forward"

curl -sf -X POST "$BASE/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null ||
    fail "unload-model failed"
guest_gone "$GUEST_PID" || fail "guest pid $GUEST_PID still alive after unload"
[[ "$(row "['state']")" == "unloaded" ]] || fail "row not unloaded after unload"
echo "[ok] unload stopped the guest"


if [[ -n "$MLX_ID" ]]; then
    load
    [[ "$(load_code "$MLX_ID")" == "200" ]] || fail "could not load $MLX_ID beside the guest"
    guest_gone "$GUEST_PID" || fail "loading $MLX_ID left guest pid $GUEST_PID running"
    [[ "$(row "['state']")" == "unloaded" ]] || fail "loading $MLX_ID left the pack '$(row "['state']")'"
    echo "[ok] loading $MLX_ID stopped the guest"

    # An idle WebSocket Responses session holds its model for as long as it is open.
    python3 - "$PORT" "$MLX_ID" >"$WORK/ws.out" 2>&1 <<'EOF' &
import base64, os, socket, sys, time
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
s.sendall(("GET /v1/responses?model=%s HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
           "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (sys.argv[2], base64.b64encode(os.urandom(16)).decode())).encode())
print("open" if b" 101 " in s.recv(4096) else "refused", flush=True)
time.sleep(120)
EOF
    WS_PID=$!
    for _ in $(seq 1 20); do [[ -s "$WORK/ws.out" ]] && break; sleep 0.5; done
    grep -q open "$WORK/ws.out" || fail "WebSocket session on $MLX_ID did not open: $(cat "$WORK/ws.out")"
    CODE=$(load_code "$ID" 60)
    [[ "$CODE" == "503" ]] && grep -q "Model $MLX_ID is serving a request; the sushi engine swaps in when it finishes" "$WORK/load.json" &&
        grep -q "Refusing to load $ID: $MLX_ID is serving a request" "$LOG" ||
        fail "loading the pack beside a busy $MLX_ID answered $CODE: $(head -c 300 "$WORK/load.json")"
    [[ "$(row "['state']" "$MLX_ID")" == "ready" && "$(row "['state']")" == "unloaded" ]] ||
        fail "a refused swap changed the models: $MLX_ID '$(row "['state']" "$MLX_ID")', pack '$(row "['state']")'"
    kill "$WS_PID" 2>/dev/null; wait "$WS_PID" 2>/dev/null || true
    WS_PID=""
    echo "[ok] a model mid-request is never evicted for the guest: a named 503, both models unchanged"

    load
    [[ "$(row "['state']" "$MLX_ID")" == "unloaded" ]] || fail "loading the pack left $MLX_ID '$(row "['state']" "$MLX_ID")'"
    echo "[ok] loading the pack evicted $MLX_ID"
else
    echo "[skip] engine swap: set SUSHI_MLX_MODEL=<small MLX model dir>"
    load
fi

kill -9 "$GUEST_PID"
sleep 1
CODE=$(curl -s -o "$WORK/dead.json" -w '%{http_code}' "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$CHAT}")
GUEST_LOG="$HOME/.mlx-serve/logs/sushi-$PORT.log"
[[ "$CODE" == "503" ]] && grep -qF "SushiGuestExited (log: $GUEST_LOG)" "$WORK/dead.json" ||
    fail "a request to a dead guest answered $CODE: $(head -c 300 "$WORK/dead.json")"
[[ "$(row "['state']")" == "error" && "$(row "['error']")" == *"sushi-$PORT.log"* ]] ||
    fail "a dead guest's row reads '$(row "['state']")'"
echo "[ok] a guest that dies fails its entry, naming its log"

curl -sf -X POST "$BASE/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null ||
    fail "unload-model of a dead guest failed"
[[ "$(row "['state']")" == "unloaded" ]] || fail "unload left a dead guest's row '$(row "['state']")'"
load
echo "[ok] unload clears a dead guest; reloaded, guest pid $GUEST_PID"
kill -9 "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
guest_gone "$GUEST_PID" || fail "guest pid $GUEST_PID outlived a killed mlx-serve"
GUEST_PID=""
echo "[ok] the guest exits when mlx-serve is killed"

echo "[pass] sushi guest"
