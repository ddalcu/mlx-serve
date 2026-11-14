#!/bin/bash
# HF_ENDPOINT: `mlx-serve pull` downloads from the hub the variable names (a mirror), sends HF_TOKEN to it, and
# ignores a value that is not an http(s) URL. Hermetic: a Python stub hub on a local port, a scratch HOME, no model.
#
#   ./tests/test_hf_endpoint.sh [stub port]
set -u
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"; exit 0; }
PORT="${1:-11516}"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/test_hf_endpoint.XXXXXX")
PASS=0; FAIL=0
ok() { if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1 — expected [$3], got [$2]"; fi; }

python3 - "$PORT" "$WORK/requests.log" <<'PY' &
import http.server, json, sys
FILES = {"config.json": b'{"model_type":"qwen3"}', "tokenizer.json": b"{}", "model.safetensors": b"\0" * 4096}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        with open(sys.argv[2], "a") as f:
            f.write(f"{self.path} auth={self.headers.get('Authorization', '')}\n")
        if self.path.startswith("/api/models/org/tiny/tree/main"):
            body = json.dumps([{"type": "file", "path": p, "size": len(b)} for p, b in FILES.items()]).encode()
        elif self.path.startswith("/org/tiny/resolve/main/") and self.path.rsplit("/", 1)[1] in FILES:
            body = FILES[self.path.rsplit("/", 1)[1]]
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
STUB=$!
trap 'kill "$STUB" 2>/dev/null; wait "$STUB" 2>/dev/null; rm -rf "$WORK"' EXIT
for _ in $(seq 1 50); do curl -s -m 1 "http://127.0.0.1:$PORT/api/models/org/tiny/tree/main" >/dev/null && break; sleep 0.1; done
: >"$WORK/requests.log"

echo "── pull through the endpoint ──"
HOME="$WORK/home" HF_ENDPOINT="http://127.0.0.1:$PORT/" HF_TOKEN="secret-token" "$BIN" pull org/tiny >"$WORK/pull.log" 2>&1
ok "pull exits 0" "$?" 0
DEST="$WORK/home/.mlx-serve/models/org/tiny"
ok "config.json landed" "$(cat "$DEST/config.json" 2>/dev/null)" '{"model_type":"qwen3"}'
ok "weights landed whole" "$(wc -c <"$DEST/model.safetensors" 2>/dev/null | tr -d ' ')" 4096
ok "the listing came from the endpoint" "$(grep -c '^/api/models/org/tiny/tree/main?recursive=true ' "$WORK/requests.log")" 1
ok "every file came from the endpoint" "$(grep -o '^/org/tiny/resolve/main/[a-z.]*' "$WORK/requests.log" | sort -u | wc -l | tr -d ' ')" 3
ok "the token rode every request to it" "$(grep -vc 'auth=Bearer secret-token$' "$WORK/requests.log")" 0

echo "── a value that is not a URL is ignored ──"
: >"$WORK/requests.log"
# huggingface.co is where this goes instead; the listing of a repo that does not exist fails there, quickly.
HOME="$WORK/home2" HF_ENDPOINT="127.0.0.1:$PORT" "$BIN" pull org/tiny-does-not-exist-mlx-serve-test >"$WORK/pull2.log" 2>&1
ok "the stub saw nothing" "$(wc -l <"$WORK/requests.log" | tr -d ' ')" 0

echo
echo "hf endpoint: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
