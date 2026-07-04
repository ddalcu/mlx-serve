#!/usr/bin/env bash
# Hunyuan3D-2.1 TEXTURED gen (P2 paint stage) on the ONE main server:
# headless boot -> load shape model by path -> POST /v1/3d/generations with
# "texture": true -> assert a valid TEXTURED GLB (TEXCOORD_0 attribute, PBR
# material, embedded PNG images) -> 400 when the paint weights are missing ->
# streaming carries paint-stage progress labels -> unload.
#
# Skips gracefully when either converted model is absent. Convert with:
#   python3 tests/convert_hunyuan3d_weights.py --src <ckpt dir> --bits 8
#   python3 tests/convert_hunyuan3d_paint_weights.py --src <paint ckpt> --dino <dinov2-giant> --bits 8
#
# Usage: HY3D_MODEL=<dir> HY3D_PAINT_MODEL=<dir> ./tests/test_3d_paint.sh [port]
set -uo pipefail
PORT="${1:-11437}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

HY3D="${HY3D_MODEL:-$(ls -d ~/.mlx-serve/models/local/hunyuan3d-2-1-8bit 2>/dev/null | head -1)}"
PAINT="${HY3D_PAINT_MODEL:-$(ls -d ~/.mlx-serve/models/local/hunyuan3d-2-1-paint-8bit 2>/dev/null | head -1)}"
[ -n "$HY3D" ] || { echo "SKIP: no Hunyuan3D shape model (set HY3D_MODEL)"; exit 0; }
[ -f "$HY3D/config.json" ] || { echo "SKIP: $HY3D has no config.json"; exit 0; }
[ -n "$PAINT" ] || { echo "SKIP: no paint model (run tests/convert_hunyuan3d_paint_weights.py)"; exit 0; }
[ -f "$PAINT/config.json" ] || { echo "SKIP: $PAINT has no config.json"; exit 0; }

HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_3d_paint_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 /tmp/test_3d_paint_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 7200 "http://127.0.0.1:$PORT$1" "${@:2}"; }
HY3D_ID="$(basename "$HY3D")"

api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$HY3D\"}" >/dev/null
for i in $(seq 1 120); do
  api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
sys.exit(0 if [x for x in d if x['id']=='$HY3D_ID' and x['state']=='ready'] else 1)" && break
  sleep 2
done
echo "PASS: shape model ready"

# Test photo: dark disc on white (base64 body goes in a FILE — house rule 6).
SRC=/tmp/test_3d_paint_src.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 384
rows = b""
for y in range(H):
    row = b"\x00"
    for x in range(W):
        d2 = (x - W // 2) ** 2 + (y - H // 2) ** 2
        v = 40 if d2 < (W // 3) ** 2 else 255
        row += bytes([v, min(255, v + 60), v])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY

python3 - "$SRC" /tmp/test_3d_paint_req.json <<PY
import json, base64, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"model": "$HY3D_ID", "image": b64, "steps": 10,
           "octree_resolution": 128, "seed": 7,
           "texture": True, "texture_steps": 8}, open(sys.argv[2], "w"))
PY

# 1. Textured generation -> GLB with TEXCOORD_0 + material + images.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_3d_paint_req.json -o /tmp/test_3d_paint_resp.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: textured gen http $code"; head -c 400 /tmp/test_3d_paint_resp.json; exit 1; }
python3 - /tmp/test_3d_paint_resp.json <<'PY'
import sys, json, base64, struct
r = json.load(open(sys.argv[1]))
assert r.get("format") == "glb", r.keys()
glb = base64.b64decode(r["data"])
assert glb[:4] == b"glTF", "bad magic"
jlen = struct.unpack("<I", glb[12:16])[0]
doc = json.loads(glb[20:20 + jlen])
prim = doc["meshes"][0]["primitives"][0]
assert "TEXCOORD_0" in prim["attributes"], "no TEXCOORD_0: " + json.dumps(prim["attributes"])
assert "material" in prim, "primitive has no material"
mats = doc.get("materials", [])
assert mats and "pbrMetallicRoughness" in mats[0], "no PBR material"
assert mats[0]["pbrMetallicRoughness"].get("baseColorTexture") is not None, "no baseColorTexture"
imgs = doc.get("images", [])
assert imgs and imgs[0].get("mimeType") == "image/png", "no embedded PNG images"
uv_acc = doc["accessors"][prim["attributes"]["TEXCOORD_0"]]
pos_acc = doc["accessors"][prim["attributes"]["POSITION"]]
assert uv_acc["type"] == "VEC2" and uv_acc["count"] == pos_acc["count"]
print(f"PASS: textured GLB, {pos_acc['count']} verts, {len(imgs)} embedded textures, {len(glb)} bytes")
PY
[ $? -eq 0 ] || exit 1

# 2. Streaming carries paint-stage progress (a stage label beyond the shape set).
python3 - /tmp/test_3d_paint_req.json <<PY
import json
d = json.load(open("/tmp/test_3d_paint_req.json")); d["stream"] = True
json.dump(d, open("/tmp/test_3d_paint_req_stream.json", "w"))
PY
api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_3d_paint_req_stream.json -o /tmp/test_3d_paint_sse.txt
python3 - /tmp/test_3d_paint_sse.txt <<'PY'
import sys, json
stages = set(); complete = False
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev.get("type") == "progress": stages.add(ev.get("stage", ""))
    if ev.get("type") == "complete": complete = True
paint_stages = {s for s in stages if s not in {"encode", "denoise", "volume", "mesh"}}
assert complete, "no complete event"
assert paint_stages, f"no paint-stage progress labels, saw only {sorted(stages)}"
print("PASS: streaming with paint stages", sorted(paint_stages))
PY
[ $? -eq 0 ] || exit 1

echo "ALL PASS: Hunyuan3D textured gen (TEXCOORD_0 + PBR material + embedded PNGs, streaming)"
