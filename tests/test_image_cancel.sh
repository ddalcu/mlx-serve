#!/usr/bin/env bash
# Client-disconnect cancellation for IMAGE generation (PR #155).
#
# A cancelled image request (client hung up mid-denoise) must abort at the
# next step — flux/krea/mage_flow poll progress.cancelled() like every other
# media backend — and gen.zig logs it as a cancellation, not a failure.
# Red on trees without the checks: the server burns through every remaining
# step + VAE decode + PNG encode for a peer that is gone (the log shows
# "[image] -> N PNG bytes" with nobody connected) while other requests queue
# behind the ghost.
#
# Usage: FLUX_MODEL=<dir> ./tests/test_image_cancel.sh [port]
set -uo pipefail
PORT="${1:-11299}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

FLUX="${FLUX_MODEL:-$(ls -d ~/.mlx-serve/models/Runpod/FLUX.2-klein-4B-mflux-4bit 2>/dev/null | head -1)}"
[ -n "$FLUX" ] || { echo "SKIP: no FLUX model (set FLUX_MODEL)"; exit 0; }
FLUX_ID="$(basename "$FLUX")"

LOG=/tmp/test_image_cancel_server.log
SSE=/tmp/test_image_cancel_sse.txt
BASE="http://127.0.0.1:$PORT"

"$BIN" --serve --model-dir "$(dirname "$FLUX")" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
CURL_PID=""
trap 'kill $CURL_PID $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "$BASE/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done

curl -s -m 300 "$BASE/v1/load-model" -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FLUX\"}" >/dev/null

# Streaming request with enough steps that the disconnect lands mid-loop.
: >"$SSE"
curl -sN -m 600 -X POST "$BASE/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FLUX_ID\",\"prompt\":\"a lighthouse at dusk\",\"size\":\"1024x1024\",\"steps\":24,\"stream\":true}" \
  >"$SSE" &
CURL_PID=$!

# Wait until the denoise loop is demonstrably running (>=2 progress events),
# then vanish. First request includes model warmup, so be patient.
IN_LOOP=0
for i in $(seq 1 180); do
  if [ "$(grep -c '"type":"progress"' "$SSE" 2>/dev/null)" -ge 2 ]; then IN_LOOP=1; break; fi
  kill -0 $CURL_PID 2>/dev/null || break
  sleep 1
done
[ "$IN_LOOP" = "1" ] || { echo "FAIL: never saw progress events"; tail -c 300 "$SSE"; tail -5 "$LOG"; exit 1; }
kill $CURL_PID 2>/dev/null
wait $CURL_PID 2>/dev/null
CURL_PID=""
echo "client disconnected mid-denoise"

# The server must notice and CANCEL. Completing the ghost job instead
# ("[image] -> N PNG bytes") is the bug.
VERDICT=""
for i in $(seq 1 240); do
  if grep -q '\[image\] generation cancelled' "$LOG"; then VERDICT=cancelled; break; fi
  if grep -q '\[image\] -> ' "$LOG"; then VERDICT=completed; break; fi
  sleep 1
done
case "$VERDICT" in
  cancelled) echo "PASS: disconnect mid-denoise -> generation cancelled" ;;
  completed) echo "FAIL: generation ran to completion for a disconnected client"; exit 1 ;;
  *)         echo "FAIL: neither cancelled nor completed after 240s"; tail -5 "$LOG"; exit 1 ;;
esac
grep -q '\[image\] generation failed' "$LOG" && { echo "FAIL: cancel misreported as a failure"; exit 1; }

# Server healthy, inference thread free: a small follow-up gen succeeds.
code=$(curl -s -m 300 -X POST "$BASE/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FLUX_ID\",\"prompt\":\"a green circle\",\"size\":\"512x512\",\"steps\":2}" \
  -o /tmp/test_image_cancel_after.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: follow-up gen http $code"; exit 1; }
python3 - /tmp/test_image_cancel_after.json <<'PY'
import sys, json, base64
b = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8] == bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
print(f"PASS: follow-up generation after cancel -> {len(b)} byte PNG")
PY
echo "ALL PASS"
