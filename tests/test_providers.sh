#!/bin/bash
# Guard: upstream chat providers (~/.mlx-serve/providers.json, src/providers.zig).
#
# Two Python stubs stand in for providers: one lists models on /v1/models,
# one has no model list at all (404) and relies on the declared `models`.
# The server boots headless with NO model; everything it answers about
# `<id>@<provider>` is proxied. HOME is redirected so the real config is
# never touched.
#
# Usage: ./tests/test_providers.sh [port]

set -u

PORT=${1:-8177}
BASE="http://127.0.0.1:$PORT"
PASS=0; FAIL=0; TOTAL=0

if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"; exit 1
fi
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "FAIL: python3 is required"; exit 1; }

WORK=$(mktemp -d ~/claude-tmp/providers-test.XXXXXX 2>/dev/null || mktemp -d)
FAKE_HOME="$WORK/home"; mkdir -p "$FAKE_HOME/.mlx-serve" "$WORK/models"
STUB_PORT=$((PORT + 1)); NOLIST_PORT=$((PORT + 2))

cat > "$WORK/stub.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
PORT = int(sys.argv[1]); LISTS = sys.argv[2] == "list"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/v1/models" and LISTS:
            body = json.dumps({"object": "list", "data": [{"id": "stub-model", "object": "model", "context_length": 32000}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers(); self.wfile.write(b"no such route")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0")); req = json.loads(self.rfile.read(n) or b"{}")
        auth = self.headers.get("Authorization", "")
        if self.path != "/v1/chat/completions":
            self.send_response(404); self.end_headers(); return
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            def chunk(s):
                b = s.encode(); self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
            chunk("data: " + json.dumps({"choices": [{"delta": {"content": "hi from " + req.get("model", "?")}}]}) + "\n\n")
            chunk("data: [DONE]\n\n"); self.wfile.write(b"0\r\n\r\n")
        else:
            body = json.dumps({"model": req.get("model"), "auth": auth,
                               "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
PY

start_stub() { python3 "$WORK/stub.py" "$STUB_PORT" list >/dev/null 2>&1 & STUB_PID=$!; }
start_stub
python3 "$WORK/stub.py" "$NOLIST_PORT" nolist >/dev/null 2>&1 & NOLIST_PID=$!

cat > "$FAKE_HOME/.mlx-serve/providers.json" <<EOF
[
  {"name":"stub","url":"http://127.0.0.1:$STUB_PORT/v1/","api_key":"literal-key","api_key_env":"MLX_TEST_PROVIDER_KEY"},
  {"name":"nolist","url":"http://127.0.0.1:$NOLIST_PORT/v1","models":["declared-a","declared-b"]},
  {"name":"off","url":"http://127.0.0.1:1/v1","enabled":false},
  {"name":"bare","url":"http://127.0.0.1:$STUB_PORT"},
  {"name":"me","url":"http://localhost:$PORT/v1"}
]
EOF

MLX_TEST_PROVIDER_KEY=env-key HOME="$FAKE_HOME" ./zig-out/bin/mlx-serve serve --port $PORT --host 127.0.0.1 \
    --log-level info --model-dir "$WORK/models" >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID $STUB_PID $NOLIST_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    [ "$FAIL" -eq 0 ] && rm -rf "$WORK" || echo "artifacts: $WORK"
}
trap cleanup EXIT

for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    [ "$i" -eq 60 ] && { echo "FAIL: server did not start"; cat "$WORK/server.log"; exit 1; }
    sleep 1
done

check() { TOTAL=$((TOTAL + 1)); if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"; else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi; }
wait_rows() { # wait_rows <jq filter> <expected>
    for i in $(seq 1 30); do
        got=$(curl -s "$BASE/v1/models" | jq -r "$1"); [ "$got" = "$2" ] && return 0; sleep 0.5
    done; return 1
}

echo "=== providers ==="
echo "[1] listed + declared rows appear, disabled provider does not"
wait_rows '[.data[] | select(.provider=="stub") | .id] | join(",")' "stub-model@stub" && r=PASS || r=FAIL
check "listed provider row is stub-model@stub" $r "$(curl -s $BASE/v1/models | jq -c '[.data[].id]')"
row=$(curl -s "$BASE/v1/models" | jq -c '.data[] | select(.id=="stub-model@stub")')
[ "$(echo "$row" | jq -r '.context_length')" = "32000" ] && [ "$(echo "$row" | jq -r '.meta.context_length')" = "32000" ] && r=PASS || r=FAIL
check "context_length twinned top-level + meta" $r "$row"
wait_rows '[.data[] | select(.provider=="nolist") | .id] | sort | join(",")' "declared-a@nolist,declared-b@nolist" && r=PASS || r=FAIL
check "no-/v1/models provider falls back to declared rows" $r
[ "$(curl -s $BASE/v1/models | jq '[.data[] | select(.provider=="off")] | length')" = "0" ] && r=PASS || r=FAIL
check "disabled provider lists nothing" $r

wait_rows '[.data[] | select(.provider=="bare") | .id] | join(",")' "stub-model@bare" && r=PASS || r=FAIL
check "a bare host:port URL is probed at /v1 and adopted" $r "$(grep 'bare' "$WORK/server.log" | head -3)"
[ "$(curl -s $BASE/v1/providers | jq -r '.providers[] | select(.name=="bare") | .url')" = "http://127.0.0.1:$STUB_PORT/v1" ] && r=PASS || r=FAIL
check "status reports the adopted /v1 base" $r
grep -q 'skipping "me": http://localhost:'"$PORT"'/v1 is this server' "$WORK/server.log" && [ "$(curl -s $BASE/v1/providers | jq '[.providers[] | select(.name=="me")] | length')" = "0" ] && r=PASS || r=FAIL
check "this server as its own provider is refused by name" $r "$(grep '"me"' "$WORK/server.log")"

echo "[2] chat completions proxied with the env key, model suffix stripped"
resp=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stub-model@stub","messages":[{"role":"user","content":"hi"}]}')
[ "$(echo "$resp" | jq -r '.model')" = "stub-model" ] && r=PASS || r=FAIL
check "upstream saw the bare model id" $r "$resp"
[ "$(echo "$resp" | jq -r '.auth')" = "Bearer env-key" ] && r=PASS || r=FAIL
check "api_key_env outranks the literal key" $r "$resp"
# Any bare id reaches the provider — the probe is a snapshot, the provider 404s its own unknowns.
resp=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"unlisted@stub","messages":[]}')
[ "$(echo "$resp" | jq -r '.model')" = "unlisted" ] && r=PASS || r=FAIL
check "unlisted model on a known provider is forwarded, not 404'd here" $r "$resp"

echo "[3] streaming relays de-chunked SSE"
stream=$(curl -s -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"declared-a@nolist","messages":[],"stream":true}')
echo "$stream" | grep -q 'hi from declared-a' && echo "$stream" | grep -q '^data: \[DONE\]' && r=PASS || r=FAIL
check "SSE body arrives with [DONE]" $r "$stream"
head=$(curl -s -N -D - -o /dev/null "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"declared-a@nolist","messages":[],"stream":true}')
echo "$head" | grep -qi '^content-type: text/event-stream' && ! echo "$head" | grep -qi 'transfer-encoding: chunked' && r=PASS || r=FAIL
check "our head: upstream content-type, no forwarded chunked framing" $r "$head"

echo "[4] other surfaces refuse by name"
code=$(curl -s -o "$WORK/msg.json" -w '%{http_code}' "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d '{"model":"stub-model@stub","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}')
[ "$code" = "400" ] && grep -q 'chat/completions only' "$WORK/msg.json" && r=PASS || r=FAIL
check "/v1/messages on a provider model is a named 400" $r "$code $(cat $WORK/msg.json)"

echo "[5] status + reload"
st=$(curl -s "$BASE/v1/providers")
[ "$(echo "$st" | jq -r '.providers[] | select(.name=="stub") | .up')" = "true" ] && [ "$(echo "$st" | jq -r '.providers[] | select(.name=="nolist") | .models')" = "2" ] && r=PASS || r=FAIL
check "GET /v1/providers reports up + model counts" $r "$st"
kill $STUB_PID; wait $STUB_PID 2>/dev/null
curl -s -X POST "$BASE/v1/providers/reload" >/dev/null
wait_rows '[.data[] | select(.provider=="stub")] | length' "0" && r=PASS || r=FAIL
check "a dead provider drops its rows after reload" $r
start_stub; sleep 0.5
curl -s -X POST "$BASE/v1/providers/reload" >/dev/null
wait_rows '[.data[] | select(.provider=="stub") | .id] | join(",")' "stub-model@stub" && r=PASS || r=FAIL
check "rows return once the provider is back" $r
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x@nobody","messages":[]}')
[ "$code" != "200" ] && ! grep -q 'proxy "x"' "$WORK/server.log" && r=PASS || r=FAIL
check "an unknown @suffix is not a provider (never proxied; the local no-model error stands)" $r "$code"
grep -q '\[providers\] stub: up, 1 models (listed)' "$WORK/server.log" && grep -q '\[providers\] nolist: up, 2 models (declared)' "$WORK/server.log" && r=PASS || r=FAIL
check "log names listed vs declared" $r "$(grep providers "$WORK/server.log" | head -5)"

echo; echo "$PASS/$TOTAL passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
