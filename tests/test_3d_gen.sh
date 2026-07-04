#!/usr/bin/env bash
# Hunyuan3D-2.1 shape gen on the ONE main server: headless boot -> load the 3D
# model by absolute path -> /v1/models shows the "3d" capability -> POST
# /v1/3d/generations with a photo -> assert a valid GLB (magic, version, JSON
# chunk parses, nonzero vertices) -> coexist with a chat model -> unload.
# Proves the fourth modality slot (.mesh) routes end to end.
#
# Skips gracefully when no converted model is present. Convert with:
#   python3 tests/convert_hunyuan3d_weights.py --src <ckpt dir> --bits 8
#
# Usage: HY3D_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_3d_gen.sh [port]
set -uo pipefail
PORT="${1:-11431}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

HY3D="${HY3D_MODEL:-$(ls -d ~/.mlx-serve/models/local/hunyuan3d-2-1-8bit 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$HY3D" ] || { echo "SKIP: no Hunyuan3D model (set HY3D_MODEL to a converted dir)"; exit 0; }
[ -f "$HY3D/config.json" ] || { echo "SKIP: $HY3D has no config.json (run tests/convert_hunyuan3d_weights.py)"; exit 0; }

# Headless: --model-dir anywhere; the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_3d_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_3d_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 3600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
HY3D_ID="$(basename "$HY3D")"

# 1. Load by absolute path -> ready with "3d" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$HY3D\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$HY3D_ID' and x['state']=='ready' and '3d' in x.get('capabilities',[])]
assert m, 'Hunyuan3D not ready with 3d cap: '+json.dumps(d)
print('PASS: load-model by path -> 3D model ready, capabilities', m[0]['capabilities'])
" || { echo "FAIL: ready 3D model missing '3d' capability"; exit 1; }

# 2. Build a test photo: dark disc centered on white (an unambiguous subject).
SRC=/tmp/test_3d_src.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 384
rows = b""
for y in range(H):
    row = bytearray([0])
    for x in range(W):
        d2 = (x - W // 2) ** 2 + (y - H // 2) ** 2
        v = 40 if d2 < (W // 3) ** 2 else 255
        row += bytes([v, v, v])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY
python3 - "$SRC" /tmp/test_3d_req.json <<PY
import json, base64, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"model": "$HY3D_ID", "image": b64, "steps": 15,
           "octree_resolution": 128, "seed": 7}, open(sys.argv[2], "w"))
PY

# 3. Generate (small res + few steps -> smoke, not quality) -> valid GLB.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_3d_req.json -o /tmp/test_3d_resp.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: 3d gen http $code"; head -c 300 /tmp/test_3d_resp.json; exit 1; }
python3 - /tmp/test_3d_resp.json <<'PY'
import sys, json, base64, struct
r = json.load(open(sys.argv[1]))
assert r.get("format") == "glb", f"format={r.get('format')}"
b = base64.b64decode(r["data"])
magic, version, total = struct.unpack("<III", b[:12])
assert magic == 0x46546C67, hex(magic)
assert version == 2, version
assert total == len(b), (total, len(b))
jlen, jtype = struct.unpack("<II", b[12:20])
assert jtype == 0x4E4F534A
g = json.loads(b[20:20 + jlen])
pos_acc = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
assert pos_acc["count"] > 0, "empty mesh"
print(f"PASS: /v1/3d/generations -> {len(b)} byte GLB, {pos_acc['count']} vertices")
PY

# 4. Server survives the gen; missing image is a 400.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after 3D gen"; exit 1; }
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HY3D_ID\",\"steps\":5}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: missing image returned $code (want 400)"; exit 1; }
echo "PASS: missing 'image' -> 400"

# 5. Streaming: SSE progress events + a complete event carrying the GLB.
# (Request body via a FILE — an inline $(python3 …) command substitution as a
# ~150 KB -d argv blob is quoting-fragile and produced an empty curl run.)
python3 - "$SRC" /tmp/test_3d_stream_req.json <<PY
import json, base64, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"model": "$HY3D_ID", "image": b64, "steps": 8,
           "octree_resolution": 128, "seed": 7, "stream": True}, open(sys.argv[2], "w"))
PY
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_3d_stream_req.json -o /tmp/test_3d_stream.txt -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream 3d gen http $code"; exit 1; }
grep -q '"type":"complete"' /tmp/test_3d_stream.txt || { echo "FAIL: no complete event in stream"; exit 1; }
echo "PASS: streaming -> SSE complete event"

# 6. Coexistence with a chat model.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while 3D model resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with 3D model also resident"
fi

# 7. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$HY3D_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$HY3D_ID']
assert m and m[0]['state']=='unloaded', '3D model should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> 3D model unloaded (stub retained)')
"

echo "ALL PASS: Hunyuan3D shape gen (headless boot, load->gen->unload, GLB validity, streaming, coexistence)"
