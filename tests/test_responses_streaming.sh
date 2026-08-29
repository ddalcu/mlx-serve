#!/bin/bash
# Integration test: /v1/responses streaming emits deltas incrementally.
#
# Regression for the bug surfaced by Local LLM Bench run #6 where
# /v1/responses with stream=true would generate the full output then
# emit a single response.output_text.delta with the entire text. That
# made TTFT == total generation time and produced absurd decode tok/s
# (e.g. 1.7M tok/s) because the bench measured deltas-per-second.
#
# Real streaming should emit many delta events spaced over the decode
# window — mirroring what /v1/chat/completions and /v1/messages do.
#
# Usage: ./tests/test_responses_streaming.sh [model_dir] [port]

set -u

MODEL_DIR=${1:-~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: ./zig-out/bin/mlx-serve not built"
    exit 1
fi

echo "=== /v1/responses Streaming Increment Test ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

echo "Starting server..."
./zig-out/bin/mlx-serve --model "$MODEL_DIR" --serve --port $PORT --log-level info \
    >/tmp/mlx-serve-stream-test.log 2>&1 &
SERVER_PID=$!
sleep 2

for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FAIL: Server did not start within 30s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done
echo ""

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

run_test() {
    local name="$1"
    local result="$2"
    local detail="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name — $detail"
    fi
}

# Capture an SSE stream with relative wall-clock timestamps per line.
# Output lines: <ms_since_request_start>\t<sse_line>
sse_with_timestamps() {
    local body="$1"
    python3 - "$BASE" "$body" <<'PY'
import sys, json, time, urllib.request
base = sys.argv[1]
body = sys.argv[2].encode()
req = urllib.request.Request(
    base + "/v1/responses",
    data=body,
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    method="POST",
)
start = time.monotonic()
with urllib.request.urlopen(req, timeout=120) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        ms = int((time.monotonic() - start) * 1000)
        sys.stdout.write(f"{ms}\t{line}\n")
        sys.stdout.flush()
PY
}

# ── Test A: text streaming emits multiple output_text.delta events ──
echo "--- Test A: incremental output_text.delta events ---"
BODY='{"model":"mlx-serve","input":"Count from 1 to 20 separated by commas.","max_output_tokens":80,"temperature":0,"stream":true}'
SSE=$(sse_with_timestamps "$BODY")

DELTA_COUNT=$(echo "$SSE" | grep -c $'\tdata: {"type":"response.output_text.delta"' || true)
echo "  observed output_text.delta events: $DELTA_COUNT"
if [ "$DELTA_COUNT" -ge 3 ]; then
    run_test "incremental output_text.delta (>=3)" "PASS" ""
else
    run_test "incremental output_text.delta (>=3)" "FAIL" "got $DELTA_COUNT (expected ≥3 for ~20-token output)"
fi

# Span between first and last delta should reflect actual decode time.
# With buffered emission, all deltas land within a few ms of each other
# at the end of generation; with real streaming they span ≥100ms.
SPAN_MS=$(echo "$SSE" | python3 -c '
import sys
firsts = []
for line in sys.stdin:
    parts = line.split("\t", 1)
    if len(parts) != 2: continue
    ms, payload = parts
    if "response.output_text.delta" in payload:
        firsts.append(int(ms))
if len(firsts) < 2:
    print(0)
else:
    print(firsts[-1] - firsts[0])
')
echo "  span(first delta → last delta): ${SPAN_MS}ms"
if [ "$SPAN_MS" -ge 100 ]; then
    run_test "deltas spread across decode window (>=100ms)" "PASS" ""
else
    run_test "deltas spread across decode window (>=100ms)" "FAIL" "span=${SPAN_MS}ms (deltas bunched — buffered emission)"
fi
echo ""

# ── Test B: reasoning streaming emits multiple summary_text.delta events ──
echo "--- Test B: incremental reasoning_summary_text.delta events ---"
BODY='{"model":"mlx-serve","input":"What is 12 + 34? Think step by step.","reasoning":{"effort":"medium"},"max_output_tokens":256,"temperature":0,"stream":true}'
SSE=$(sse_with_timestamps "$BODY")

R_DELTA_COUNT=$(echo "$SSE" | grep -c $'\tdata: {"type":"response.reasoning_summary_text.delta"' || true)
T_DELTA_COUNT=$(echo "$SSE" | grep -c $'\tdata: {"type":"response.output_text.delta"' || true)
echo "  observed reasoning_summary_text.delta: $R_DELTA_COUNT"
echo "  observed output_text.delta:            $T_DELTA_COUNT"

# We accept either reasoning OR text emitted incrementally — model may
# decide to skip reasoning. The combined count must show streaming.
COMBINED=$((R_DELTA_COUNT + T_DELTA_COUNT))
if [ "$COMBINED" -ge 3 ]; then
    run_test "incremental reasoning/text deltas (>=3 combined)" "PASS" ""
else
    run_test "incremental reasoning/text deltas (>=3 combined)" "FAIL" "got $COMBINED (expected ≥3)"
fi
echo ""

# ── Test C: HTTP SSE terminates with the `data: [DONE]` sentinel ──
# OpenAI ends every Responses HTTP SSE stream with the same terminal
# `data: [DONE]` sentinel as chat completions, after `response.completed`.
# Generic SSE middleware (LiteLLM-class proxies) keys stream end off it.
# The WebSocket transport must NOT get one (its terminator is the
# response.completed event itself) — this test covers HTTP only.
echo "--- Test C: terminal data: [DONE] sentinel (HTTP SSE) ---"
BODY='{"model":"mlx-serve","input":"Say hi.","max_output_tokens":16,"temperature":0,"stream":true}'
SSE=$(sse_with_timestamps "$BODY")

LAST_DATA=$(echo "$SSE" | grep $'\tdata: ' | tail -1 | cut -f2-)
COMPLETED_COUNT=$(echo "$SSE" | grep -c $'\tdata: {"type":"response.completed"' || true)
echo "  last data frame: ${LAST_DATA:0:60}"
if [ "$LAST_DATA" = "data: [DONE]" ]; then
    run_test "stream terminates with data: [DONE]" "PASS" ""
else
    run_test "stream terminates with data: [DONE]" "FAIL" "last data frame was: ${LAST_DATA:0:80}"
fi
if [ "$COMPLETED_COUNT" -ge 1 ]; then
    run_test "response.completed precedes [DONE]" "PASS" ""
else
    run_test "response.completed precedes [DONE]" "FAIL" "no response.completed event seen"
fi
echo ""

# ── Test D: background mode is honestly rejected ──
# The server has no queue/poll machinery. Accepting `background:true` and
# running synchronously returns status "completed" on a request the caller
# expects to poll — a silent lie flagged by llmprobe. A clean 400 is the
# honest answer (probes report it as "unsupported", not failed).
echo "--- Test D: background:true returns 400 ---"
BG_BODY='{"model":"mlx-serve","input":"Say hi.","max_output_tokens":16,"background":true}'
BG_CODE=$(curl -s -o /tmp/responses_bg_test.out -w '%{http_code}' \
    -X POST "$BASE/v1/responses" -H 'Content-Type: application/json' -d "$BG_BODY")
BG_BODY_OUT=$(cat /tmp/responses_bg_test.out)
echo "  status: $BG_CODE"
if [ "$BG_CODE" = "400" ] && echo "$BG_BODY_OUT" | grep -q "background"; then
    run_test "background:true rejected with 400 naming background" "PASS" ""
else
    run_test "background:true rejected with 400 naming background" "FAIL" "status=$BG_CODE body=${BG_BODY_OUT:0:120}"
fi
echo ""

# ── Test E: the WS transport NEVER gets a [DONE] frame; chaining survives ──
# The mirror of Test C. Over WebSocket the per-response terminator is the
# `response.completed` event; the compliance suite advances turns on it, so a
# trailing [DONE] frame would be misread as the NEXT turn's marker and kill
# chained sessions. Two chained turns on one socket must complete with zero
# [DONE] frames.
echo "--- Test E: WS transport — chained turns, zero [DONE] frames ---"
WS_OUT=$(python3 - "$PORT" <<'PY'
import base64, json, os, socket, sys

sock = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=120)
key = base64.b64encode(os.urandom(16)).decode()
sock.sendall((
    f"GET /v1/responses HTTP/1.1\r\nHost: 127.0.0.1:{sys.argv[1]}\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
).encode())
hdr = b""
while b"\r\n\r\n" not in hdr:
    hdr += sock.recv(1024)
assert b"101" in hdr.split(b"\r\n")[0], hdr

def send_frame(payload: bytes):
    mask = os.urandom(4)
    header = bytearray([0x81])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    else:
        header.append(0x80 | 126)
        header += n.to_bytes(2, "big")
    sock.sendall(bytes(header) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def run_turn(body):
    send_frame(json.dumps(body).encode())
    saw_done = False
    while True:
        b0, b1 = read_exact(2)
        n = b1 & 0x7F
        if n == 126:
            n = int.from_bytes(read_exact(2), "big")
        elif n == 127:
            n = int.from_bytes(read_exact(8), "big")
        payload = read_exact(n) if n else b""
        if (b0 & 0x0F) == 0x8:
            raise ConnectionError("server closed mid-turn")
        if (b0 & 0x0F) != 0x1:
            continue
        text = payload.decode("utf-8", errors="replace")
        if "[DONE]" in text:
            saw_done = True
        try:
            ev = json.loads(text)
        except ValueError:
            continue
        if ev.get("type") == "response.completed":
            return ev["response"]["id"], saw_done
        if ev.get("type") in ("response.failed", "response.incomplete"):
            raise RuntimeError(ev.get("type"))

turn = {"type": "response.create", "model": "mlx-serve", "input": "Say hi.",
        "max_output_tokens": 16, "temperature": 0}
id1, done1 = run_turn(turn)
turn2 = dict(turn, input="Say bye.", previous_response_id=id1)
id2, done2 = run_turn(turn2)
print(f"ws {0 if (done1 or done2) else 1} chained={bool(id1 and id2)} done_frames={done1 or done2}")
PY
) || WS_OUT="ws 0 exception"
echo "  $WS_OUT"
if [ "$(echo "$WS_OUT" | awk '/^ws/{print $2}')" = "1" ]; then
    run_test "WS: two chained turns complete, zero [DONE] frames" "PASS" ""
else
    run_test "WS: two chained turns complete, zero [DONE] frames" "FAIL" "$WS_OUT"
fi
echo ""

echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
