#!/usr/bin/env bash
# Chat keeps flowing while a media job runs (`GenYield`). One server, a small
# chat model + FLUX klein:
#   [1] a chat sent mid-gen finishes BEFORE the gen does,
#   [2] the gen's image is byte-identical to a solo run of the same seed,
#   [3] the log carries `[gen-yield] engaged` (an arm that never yields is a silent no-op),
#   [4] a gen whose client hangs up frees the server: the next chat answers within a step.
#   [5] (Hunyuan3D + paint present) every chat sent while a textured res=320 job
#       runs answers promptly (its CPU stages run on a worker).
# Usage: CHAT_MODEL=<dir> FLUX_MODEL=<dir> [HY3D_MODEL=<dir>] ./tests/test_gen_chat_interleave.sh [port]
set -uo pipefail
PORT="${1:-11461}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
M=~/.mlx-serve/models
CHAT="${CHAT_MODEL:-$M/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
FLUX="${FLUX_MODEL:-$M/Runpod/FLUX.2-klein-4B-mflux-4bit}"
HY3D="${HY3D_MODEL:-$M/ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit}"
[ -f "$CHAT/config.json" ] || { echo "SKIP: no chat model (set CHAT_MODEL)"; exit 0; }
[ -d "$FLUX" ] || { echo "SKIP: no FLUX model (set FLUX_MODEL)"; exit 0; }

TMP=$(mktemp -d)
LOG="$TMP/server.log"
"$BIN" --serve --port "$PORT" --log-level debug >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done
api() { curl -s -m 1800 "http://127.0.0.1:$PORT$1" "${@:2}"; }
now() { python3 -c 'import time; print(time.time())'; }
load() {
  local code
  code=$(api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$1\"}" -o /dev/null -w "%{http_code}")
  [ "$code" = "200" ] || { echo "FAIL: load $1 http $code"; exit 1; }
}
load "$CHAT"
load "$FLUX"
echo "PASS: chat + image models ready"

chat() { # $1 = output file; prints wall seconds
  local t0 t1
  t0=$(now)
  api /v1/chat/completions -H 'Content-Type: application/json' -o "$1" -d "{\"model\":\"$CHAT\",\"messages\":[{\"role\":\"user\",\"content\":\"Name three colors.\"}],\"max_tokens\":32,\"temperature\":0,\"enable_thinking\":false}"
  t1=$(now)
  python3 -c "print($t1 - $t0)"
}
chat_ok() { python3 -c "import json,sys; d=json.load(open('$1')); assert d['choices'][0]['message']['content'].strip(), d" 2>/dev/null; }
IMG_REQ="{\"model\":\"$FLUX\",\"prompt\":\"a lighthouse on a cliff at dusk\",\"size\":\"1024x1024\",\"steps\":16,\"seed\":7}"
b64() { python3 -c "import json; print(json.load(open('$1'))['data'][0]['b64_json'])"; }

# Solo reference (also warms the chat model's first-call JIT).
chat "$TMP/warm.json" >/dev/null
t0=$(now)
api /v1/images/generations -H 'Content-Type: application/json' -d "$IMG_REQ" -o "$TMP/solo.json"
SOLO=$(python3 -c "print($(now) - $t0)")
b64 "$TMP/solo.json" >"$TMP/solo.b64" || { echo "FAIL: solo gen"; head -c 300 "$TMP/solo.json"; exit 1; }
echo "solo gen: ${SOLO}s"

# [1][2] A chat sent mid-gen finishes first; the image does not change.
( api /v1/images/generations -H 'Content-Type: application/json' -d "$IMG_REQ" -o "$TMP/mixed.json"; now >"$TMP/gen_end" ) &
GEN=$!
sleep "$(python3 -c "print(max(1.0, $SOLO / 4))")"
chat "$TMP/mid.json" >/dev/null
CHAT_END=$(now)
wait $GEN
chat_ok "$TMP/mid.json" || { echo "FAIL: mid-gen chat"; head -c 300 "$TMP/mid.json"; exit 1; }
python3 -c "import sys; sys.exit(0 if $CHAT_END < $(cat "$TMP/gen_end") else 1)" ||
  { echo "FAIL: [1] the chat waited for the gen (chat end $CHAT_END, gen end $(cat "$TMP/gen_end"))"; exit 1; }
echo "PASS: [1] chat finished before the gen"
b64 "$TMP/mixed.json" >"$TMP/mixed.b64" || { echo "FAIL: mixed gen"; exit 1; }
cmp -s "$TMP/solo.b64" "$TMP/mixed.b64" || { echo "FAIL: [2] interleaving changed the image"; exit 1; }
echo "PASS: [2] gen bytes == solo run"

# [3] Engagement.
grep -q "\[gen-yield\] engaged" "$LOG" || { echo "FAIL: [3] no [gen-yield] engaged line"; exit 1; }
echo "PASS: [3] [gen-yield] engaged"

# [4] Cancel: hang up mid-denoise, the next chat answers within a step.
STREAM_REQ="${IMG_REQ%\}},\"stream\":true}"
curl -sN -m 1800 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' -d "$STREAM_REQ" >"$TMP/cancel.sse" &
CUR=$!
for i in $(seq 1 600); do grep -q '"step":2' "$TMP/cancel.sse" 2>/dev/null && break; sleep 0.1; done
kill $CUR 2>/dev/null; wait $CUR 2>/dev/null
LAT=$(chat "$TMP/after.json")
chat_ok "$TMP/after.json" || { echo "FAIL: post-cancel chat"; exit 1; }
python3 -c "import sys; sys.exit(0 if $LAT < $SOLO / 2 else 1)" ||
  { echo "FAIL: [4] chat after a cancel took ${LAT}s (solo gen ${SOLO}s)"; exit 1; }
for i in $(seq 1 50); do grep -q "\[image\] generation cancelled" "$LOG" && break; sleep 0.2; done
grep -q "\[image\] generation cancelled" "$LOG" || { echo "FAIL: [4] the abandoned gen was not cancelled"; exit 1; }
echo "PASS: [4] cancelled gen freed the server (chat ${LAT}s)"

# [5] Hunyuan3D textured: chat during offloaded CPU stages.
if [ -f "$HY3D/config.json" ] && [ -f "$HY3D/paint/config.json" ]; then
  load "$HY3D"
  python3 - "$TMP/mesh.json" "$HY3D" <<'PY'
import json, base64, struct, sys, zlib
W = H = 384
rows = b"".join(b"\x00" + b"".join(bytes([40, 100, 40]) if (x - 192) ** 2 + (y - 192) ** 2 < 128 ** 2 else b"\xff\xff\xff" for x in range(W)) for y in range(H))
ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(rows)) + ch(b"IEND", b"")
json.dump({"model": sys.argv[2], "image": base64.b64encode(png).decode(), "steps": 10, "octree_resolution": 320,
           "seed": 7, "texture": True, "texture_steps": 8}, open(sys.argv[1], "w"))
PY
  ( api /v1/3d/generations -H 'Content-Type: application/json' -d @"$TMP/mesh.json" -o "$TMP/mesh_resp.json" -w "%{http_code}" >"$TMP/mesh_code" ) &
  MESH=$!
  MAXLAT=0; N=0
  while kill -0 $MESH 2>/dev/null; do
    LAT=$(chat "$TMP/m$N.json")
    kill -0 $MESH 2>/dev/null || break # answered after the job: not a mid-job sample
    chat_ok "$TMP/m$N.json" || { echo "FAIL: [5] chat during the mesh job"; exit 1; }
    MAXLAT=$(python3 -c "print(max($MAXLAT, $LAT))"); N=$((N + 1))
    sleep 1
  done
  wait $MESH
  [ "$(cat "$TMP/mesh_code")" = "200" ] || { echo "FAIL: [5] textured job http $(cat "$TMP/mesh_code")"; exit 1; }
  grep -q "decimated [0-9]* -> [0-9]* faces" "$LOG" || { echo "FAIL: [5] the paint stage never ran"; exit 1; }
  [ "$N" -gt 0 ] || { echo "FAIL: [5] no chat landed during the mesh job"; exit 1; }
  python3 -c "import sys; sys.exit(0 if $MAXLAT < 30 else 1)" ||
    { echo "FAIL: [5] a chat during the mesh job took ${MAXLAT}s"; exit 1; }
  echo "PASS: [5] $N chats during the textured job, slowest ${MAXLAT}s"
else
  echo "SKIP: [5] no Hunyuan3D shape + paint pack"
fi
echo "ALL PASS: chat interleaves with media generation"
