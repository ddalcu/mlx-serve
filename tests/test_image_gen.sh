#!/usr/bin/env bash
# Native FLUX.2 image endpoint smoke test.
# Usage: FLUX_MODEL=<dir> ./tests/test_image_gen.sh [port]
set -uo pipefail
PORT="${1:-11399}"
MODEL="${FLUX_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--Runpod--FLUX.2-klein-4B-mflux-4bit/snapshots/* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no FLUX model (set FLUX_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

"$BIN" --model "$MODEL" --serve --port "$PORT" >/tmp/test_image_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
# wait for /health (model load is heavy)
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 /tmp/test_image_server.log; exit 1; }
  sleep 2
done

OUT=/tmp/test_image_gen.json
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a wooden table","size":"1024x1024"}' -o "$OUT" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; head -c 300 "$OUT"; exit 1; }
# decode b64 PNG, check magic + dims
python3 - "$OUT" <<'PY'
import sys, json, base64, struct
d=json.load(open(sys.argv[1]))
b=base64.b64decode(d["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
print(f"PASS: /v1/images/generations -> {len(b)} byte PNG {w}x{h}")
assert w==1024 and h==1024, f"bad dims {w}x{h}"
PY
