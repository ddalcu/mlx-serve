#!/usr/bin/env bash
# MiniMax-H3 residency: the text encoder and DiT stay loaded between requests while memory allows.
#
#  [1] a second identical request HITS the text-encoder, DiT and Turbo-LoRA caches (counted from the log, never inferred from timing)
#      and returns the same pixels
#  [2] a plain request after a Turbo one carries no Turbo residue: it matches a plain request on a
#      fresh server (the resident DiT's LoRA slots are cleared between requests)
#  [3] MLX_SERVE_H3_RESIDENT=0 runs the staged plan: no cache hits
#  [4] two keyframe requests in a row: the second reuses the resident text encoder's vision tower
#      (the weights map is not reopened) and answers 200 — the server does not die
#
# Usage: [H3_MODEL=<dir>] ./tests/test_h3_resident.sh [port]
set -uo pipefail
PORT="${1:-11363}"
MODEL="${H3_MODEL:-$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3 pack at $MODEL (set H3_MODEL)"; exit 0; }
[ -f "$MODEL/turbo_lora.safetensors" ] || { echo "SKIP: pack has no turbo_lora.safetensors"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
rc=0
SRV=""
trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT

boot() { # <log> [env assignment]
  [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }
  env ${2:-X=1} "$BIN" --model "$MODEL" --serve --port "$PORT" >"$1" 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$1"; exit 1; }
    sleep 1
  done
}

# gen <turbo true|false> <steps> -> "<sha1> <seconds>"
gen() {
  python3 - "$PORT" "$1" "$2" <<'PY'
import base64, hashlib, json, sys, time, urllib.request
port, turbo, steps = sys.argv[1], sys.argv[2] == "true", int(sys.argv[3])
prompt = ("integrated_multimodal_description:\nA woman at a kitchen table in warm morning light, looking into the camera.\n\n"
          "overall_soundscape:\nSoft room tone.\n\nnon_diegetic_music:\nN/A")
body = {"prompt": prompt, "width": 640, "height": 384, "num_frames": 22, "steps": steps, "seed": 7, "turbo": turbo, "fast": False}
t = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/video/generations", json.dumps(body).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=1800))
print(hashlib.sha1(base64.b64decode(d["data"])).hexdigest(), f"{time.time() - t:.1f}")
PY
}

hits() { grep -c "$2" "$1"; }

echo "[1] resident hits + determinism"
boot /tmp/h3_res_a.log
read -r sha1 t1 < <(gen true 4)
read -r sha2 t2 < <(gen true 4)
te_hits=$(hits /tmp/h3_res_a.log "text encoder: resident hit")
dit_hits=$(hits /tmp/h3_res_a.log "dit: resident hit")
lora_hits=$(hits /tmp/h3_res_a.log "lora: resident hit")
if [ "$te_hits" = 1 ] && [ "$dit_hits" = 1 ] && [ "$lora_hits" = 1 ]; then echo "PASS: second request hit the text encoder, DiT and Turbo LoRA caches (${t1}s -> ${t2}s)"; else echo "FAIL: cache hits te=$te_hits dit=$dit_hits lora=$lora_hits (want 1 each)"; rc=1; fi
[ "$sha1" = "$sha2" ] && echo "PASS: the resident request returns the same pixels" || { echo "FAIL: pixels differ between request 1 and 2"; rc=1; }

echo "[2] no Turbo residue in a plain request"
read -r plain_after < <(gen false 6)
boot /tmp/h3_res_b.log
read -r plain_fresh < <(gen false 6)
[ "${plain_after%% *}" = "${plain_fresh%% *}" ] && echo "PASS: a plain request after Turbo equals a plain request on a fresh server" || { echo "FAIL: LoRA residue: ${plain_after%% *} vs ${plain_fresh%% *}"; rc=1; }

echo "[3] kill switch"
boot /tmp/h3_res_c.log MLX_SERVE_H3_RESIDENT=0
gen true 4 >/dev/null; gen true 4 >/dev/null
n=$(hits /tmp/h3_res_c.log "resident hit")
[ "$n" = 0 ] && echo "PASS: MLX_SERVE_H3_RESIDENT=0 runs the staged plan" || { echo "FAIL: $n cache hits with residency off"; rc=1; }
echo "[4] keyframes on a resident text encoder"
if ! grep -q '"fl2va"' "$MODEL/config.json"; then
  echo "SKIP: pack declares no fl2va task"
else
  boot /tmp/h3_res_d.log
  IMG=/tmp/h3_res_frame.png
  python3 - "$IMG" <<'PY'
import sys, struct, zlib
W, H = 256, 256
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
raw = bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        v = 20 if x < W // 2 else 235
        raw += bytes((v, v, v))
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
  B64=$(base64 < "$IMG" | tr -d '\n')
  python3 -c "import json,sys;json.dump({'prompt':'a static abstract scene of two solid color fields','num_frames':5,'width':256,'height':256,'steps':4,'seed':3,'turbo':True,'fast':False,'first_frame_image':sys.argv[1]}, open('/tmp/h3_res_kf_req.json','w'))" "$B64"
  kf() { curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' --data @/tmp/h3_res_kf_req.json -o /dev/null -w "%{http_code}"; }
  c1=$(kf); c2=$(kf)
  te_hits=$(hits /tmp/h3_res_d.log "text encoder: resident hit")
  if [ "$c1" = 200 ] && [ "$c2" = 200 ] && [ "$te_hits" = 1 ] && kill -0 $SRV 2>/dev/null; then echo "PASS: the second keyframe request hit the resident text encoder (http $c1/$c2)"; else echo "FAIL: keyframe requests http $c1/$c2, te hits $te_hits, server alive=$(kill -0 $SRV 2>/dev/null && echo yes || echo no)"; tail -5 /tmp/h3_res_d.log; rc=1; fi
fi
exit $rc
