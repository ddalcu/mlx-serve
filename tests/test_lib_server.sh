#!/bin/bash
# Hermetic tests for tests/lib/server.sh and its Python twin tests/lib/server.py: the /health wait
# (200 ready, 503 loading keeps waiting, a dead pid stops it), the log-line wait, and srv_stop.
# Python stub servers on ports 11391-11392, no model, no mlx-serve binary.
#
# Run: bash tests/test_lib_server.sh   (a failure means a helper's verdict changed; every live script inherits it)
set -u
cd "$(dirname "$0")"
source lib/server.sh

PASS=0; FAIL=0
ok() { # name  actual  expected
    if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL $1 — expected [$3], got [$2]"; fi
}

TMP="$(mktemp -d)"
STUBS=()
cleanup() {
    for p in ${STUBS[@]+"${STUBS[@]}"}; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    rm -rf "$TMP"
}
trap cleanup EXIT
stub() { # stub <port> <status>: /health answers <status>
    python3 -c '
import http.server, sys
code = int(sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(code); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$1" "$2" &
    STUBS+=($!)
}
OK_URL=http://127.0.0.1:11391; LOADING_URL=http://127.0.0.1:11392
for u in "$OK_URL" "$LOADING_URL" http://127.0.0.1:11393; do
    curl -s -m 1 "$u/health" >/dev/null && { echo "FAIL: ${u##*:} is already in use"; exit 1; }
done
stub 11391 200; stub 11392 503
for _ in $(seq 1 50); do curl -s -m 1 "$OK_URL/health" >/dev/null && curl -s -m 1 "$LOADING_URL/health" >/dev/null && break; sleep 0.1; done
for p in "${STUBS[@]}"; do kill -0 "$p" 2>/dev/null || { echo "FAIL: a stub server did not start"; exit 1; }; done

echo "── srv_wait_health ──"
srv_wait_health "$OK_URL" 3 0.1; ok "200 is ready" "$?" 0
srv_wait_health "$LOADING_URL" 3 0.1; ok "503 loading is not ready" "$?" 1
srv_wait_health "http://127.0.0.1:11393" 3 0.1; ok "nothing listening is not ready" "$?" 1
true & DEAD=$!; wait "$DEAD"   # exits on its own: a signalled fork can run the EXIT trap
start=$SECONDS; srv_wait_health "$LOADING_URL" 100 0.1 "$DEAD"; rc=$?
ok "a dead pid stops the wait" "$rc" 1
ok "a dead pid stops it at once" "$(( SECONDS - start < 3 ))" 1
sleep 30 & LIVE=$!
srv_wait_health "$OK_URL" 3 0.1 "$LIVE"; ok "200 with a live pid is ready" "$?" 0

echo "── srv_wait_log ──"
LOG="$TMP/server.log"; : >"$LOG"
( sleep 0.3; echo "Model ready (loaded on inference thread)" >>"$LOG" ) &
srv_wait_log "$LOG" "Model ready (loaded on inference thread)" 50 0.1 "$LIVE"; ok "the line arrives (parens literal)" "$?" 0
srv_wait_log "$LOG" "never printed" 3 0.1; ok "a missing line times out" "$?" 1
srv_wait_log "$LOG" "never printed" 100 0.1 "$DEAD"; ok "a dead pid stops the log wait" "$?" 1

echo "── srv_stop ──"
srv_stop "$LIVE"; kill -0 "$LIVE" 2>/dev/null; ok "the server is gone" "$?" 1
srv_stop ""; ok "an empty pid is a no-op" "$?" 0

echo "── lib/server.py ──"
py() { python3 -c "
import subprocess, sys
from lib.server import wait_health, stop
$1" 2>&1; }
ok "200 is ready"              "$(py "wait_health('$OK_URL', 2, interval=0.1); print('ok')")" ok
ok "503 loading times out"     "$(py "
try: wait_health('$LOADING_URL', 0.5, interval=0.1)
except TimeoutError: print('timeout')")" timeout
ok "an exited process raises"  "$(py "
p = subprocess.Popen(['true']); p.wait()
try: wait_health('$LOADING_URL', 30, p, interval=0.1)
except RuntimeError as e: print('exited')")" exited
ok "stop reaps the process"    "$(py "
p = subprocess.Popen(['sleep', '30']); stop(p); print(p.returncode is not None)")" True

echo
echo "lib/server: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
