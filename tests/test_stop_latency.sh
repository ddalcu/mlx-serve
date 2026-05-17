#!/bin/bash
# Stop-latency test: when the client cancels a streaming request mid-decode,
# the server's connection thread must notice within ~100ms and stop the
# inference for that slot. Without this, the GPU keeps generating tokens for
# many seconds — fans spin, energy burns, partial work is discarded.
#
# How we measure: tokenize the server's log finish line "<- N+M tokens
# streamed ... [client_disconnect]" — we record the client's token count at
# the kill moment and compare against the server's reported M. Anything more
# than ~1 spec-block of extra tokens flags a regression.
#
# Requires:
#   - A built mlx-serve binary
#   - A model directory
#
# Usage:
#   STOP_TEST_MODEL=/path/to/model ./tests/test_stop_latency.sh [port]

set -e

PORT=${1:-8095}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${STOP_TEST_MODEL:-${PLD_TEST_MODEL:-$HOME/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit}}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_stop_latency: model directory not found."
    echo "  Set STOP_TEST_MODEL or place an MLX checkpoint at"
    echo "  ~/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit."
    exit 0
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable."
    exit 1
fi

LOGFILE=$(mktemp)
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
"$BINARY" --model "$MODEL" --serve --port "$PORT" > "$LOGFILE" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; rm -f $LOGFILE" EXIT INT TERM

echo "== stop-latency test =="
echo "  model: $MODEL"
echo "  log:   $LOGFILE"

up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server didn't come up in 60s"
    tail -30 "$LOGFILE"
    exit 1
fi

# Per kill-at-T test: fire a long streaming request, drop the connection
# after T seconds with SO_LINGER=0 (immediate RST), then read the server
# log for the matching finish line and compute (server_total - client_count).
run_one() {
    local kill_after=$1
    : > "$LOGFILE"
    python3 - "$PORT" "$kill_after" "$LOGFILE" <<'PY'
import socket, struct, time, json, sys, subprocess, re

port = int(sys.argv[1])
kill_after = float(sys.argv[2])
logfile = sys.argv[3]

body = json.dumps({
    "model": "mlx-serve",
    "messages": [{"role": "user", "content": "Tell me a long story about a robot."}],
    "max_tokens": 2000,
    "temperature": 0.8,
    "stream": True,
}).encode()
req = (
    b"POST /v1/chat/completions HTTP/1.1\r\n"
    b"Host: 127.0.0.1:%d\r\n" % port +
    b"Content-Type: application/json\r\n"
    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
    b"Connection: close\r\n\r\n"
) + body
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", port))
s.sendall(req)

start = time.time()
client_count = 0
while True:
    elapsed = time.time() - start
    if elapsed >= kill_after: break
    s.settimeout(max(0.05, kill_after - elapsed))
    try: data = s.recv(4096)
    except socket.timeout: continue
    if not data: print(f"server closed early at t={elapsed:.2f}s"); sys.exit(1)
    client_count += data.count(b'"delta":{"content"')

kill_t = time.time()
s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
s.close()

# Wait up to 5s for server to log the finish line.
deadline = time.time() + 5
found = None
while time.time() < deadline:
    out = subprocess.run(["grep", "-a", "tokens streamed", logfile],
                        capture_output=True, text=True).stdout
    if out.strip():
        found = out.strip().splitlines()[-1]
        latency_ms = (time.time() - kill_t) * 1000
        break
    time.sleep(0.05)

if not found:
    print(f"FAIL kill_after={kill_after}s: no finish line in {logfile} within 5s")
    sys.exit(1)

m = re.search(r"<- (\d+)\+(\d+) tokens streamed.*\[(\w+)\]", found)
if not m:
    print(f"FAIL kill_after={kill_after}s: couldn't parse finish line: {found}")
    sys.exit(1)

server_total = int(m.group(2))
reason = m.group(3)
extra = server_total - client_count

print(f"  kill@{kill_after}s: client={client_count} server={server_total} extra={extra} reason={reason} log_latency={latency_ms:.0f}ms")

# Pass criteria:
#   - reason should be client_disconnect (not stop/length — model finished naturally)
#   - extra ≤ 16 tokens (≤ one spec-decode block in the worst case: drafter block_size 8 + a small buffer)
if reason != "client_disconnect":
    print(f"  FAIL: expected reason 'client_disconnect', got '{reason}' — server didn't detect the drop")
    sys.exit(1)
if extra > 16:
    print(f"  FAIL: server emitted {extra} tokens after kill (target ≤16) — disconnect-detection latency too high")
    sys.exit(1)
PY
}

# Try several kill timings — early kills exercise the first decode tick,
# later kills exercise the steady-state loop.
PASS=0
FAIL=0
for kill_at in 0.5 1.0 2.0 3.0; do
    if run_one "$kill_at"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done

echo
echo "  passed: $PASS / $((PASS+FAIL))"
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}FAIL${NC} $FAIL of $((PASS+FAIL)) kill points exceeded the latency budget"
    exit 1
fi
echo -e "${GREEN}PASS${NC} all kill points within 16-token / 5s budget"
