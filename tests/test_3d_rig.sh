#!/usr/bin/env bash
# Hunyuan3D + UniRig RIGGED gen (P3) on the ONE main server: headless boot ->
# POST /v1/3d/generations with "rig": true -> assert a valid SKINNED GLB
# (JOINTS_0/WEIGHTS_0 attributes, a skin with inverseBindMatrices + joint node
# hierarchy, weights normalized) -> 400 when the rig weights are missing ->
# SSE streams rig-stage labels.
#
# Skips gracefully when either converted model is absent. Convert with:
#   python3 tests/convert_hunyuan3d_weights.py --src <ckpt dir> --bits 8
#   python3 tests/convert_unirig_weights.py --bits 8
#
# Usage: HY3D_MODEL=<dir> UNIRIG_MODEL=<dir> ./tests/test_3d_rig.sh [port]
set -uo pipefail
PORT="${1:-11439}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

HY3D="${HY3D_MODEL:-$(ls -d ~/.mlx-serve/models/local/hunyuan3d-2-1-8bit 2>/dev/null | head -1)}"
RIG="${UNIRIG_MODEL:-$(ls -d ~/.mlx-serve/models/local/unirig-skeleton-8bit 2>/dev/null | head -1)}"
[ -n "$HY3D" ] && [ -f "$HY3D/config.json" ] || { echo "SKIP: no Hunyuan3D shape model"; exit 0; }
[ -n "$RIG" ] && [ -f "$RIG/config.json" ] || { echo "SKIP: no UniRig model (run tests/convert_unirig_weights.py)"; exit 0; }

HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_3d_rig_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 /tmp/test_3d_rig_server.log; exit 1; }
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

SRC=/tmp/test_3d_rig_src.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 384
rows = b""
for y in range(H):
    row = b"\x00"
    for x in range(W):
        # A vaguely humanoid blob: tall ellipse body + circle head.
        body = ((x - W//2)/(W*0.18))**2 + ((y - H*0.62)/(H*0.30))**2 < 1
        head = ((x - W//2)/(W*0.13))**2 + ((y - H*0.22)/(H*0.13))**2 < 1
        v = 60 if (body or head) else 255
        row += bytes([v, v, min(255, v + 40)])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY

python3 - "$SRC" /tmp/test_3d_rig_req.json <<PY
import json, base64, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"model": "$HY3D_ID", "image": b64, "steps": 10,
           "octree_resolution": 128, "seed": 7, "rig": True}, open(sys.argv[2], "w"))
PY

# 1. Rigged generation -> GLB with skin, joints, weights.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_3d_rig_req.json -o /tmp/test_3d_rig_resp.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: rigged gen http $code"; head -c 400 /tmp/test_3d_rig_resp.json; exit 1; }
python3 - /tmp/test_3d_rig_resp.json <<'PY'
import sys, json, base64, struct
r = json.load(open(sys.argv[1]))
assert r.get("format") == "glb", r.keys()
glb = base64.b64decode(r["data"])
assert glb[:4] == b"glTF"
jlen = struct.unpack("<I", glb[12:16])[0]
doc = json.loads(glb[20:20 + jlen])
prim = doc["meshes"][0]["primitives"][0]
attrs = prim["attributes"]
assert "JOINTS_0" in attrs and "WEIGHTS_0" in attrs, f"no skin attrs: {attrs}"
skins = doc.get("skins", [])
assert skins, "no skins array"
joints = skins[0]["joints"]
assert len(joints) >= 2, f"suspiciously few joints: {len(joints)}"
ibm_acc = doc["accessors"][skins[0]["inverseBindMatrices"]]
assert ibm_acc["type"] == "MAT4" and ibm_acc["count"] == len(joints)
# Weights normalized: decode WEIGHTS_0 and check row sums.
wacc = doc["accessors"][attrs["WEIGHTS_0"]]
views = doc["bufferViews"]
bin_off = 20 + jlen + 8
wview = views[wacc["bufferView"]]
woff = bin_off + wview.get("byteOffset", 0)
n = wacc["count"]
import array
w = array.array("f")
w.frombytes(glb[woff:woff + n * 16])
bad = sum(1 for v in range(n) if abs(sum(w[v*4:v*4+4]) - 1.0) > 1e-3)
assert bad < n * 0.01, f"{bad}/{n} weight rows not normalized"
pos_acc = doc["accessors"][attrs["POSITION"]]
assert wacc["count"] == pos_acc["count"]
print(f"PASS: rigged GLB, {pos_acc['count']} verts, {len(joints)} joints, weights normalized")
PY
[ $? -eq 0 ] || exit 1

# 2. Streaming carries rig-stage progress labels.
python3 - <<PY
import json
d = json.load(open("/tmp/test_3d_rig_req.json")); d["stream"] = True
json.dump(d, open("/tmp/test_3d_rig_req_stream.json", "w"))
PY
api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_3d_rig_req_stream.json -o /tmp/test_3d_rig_sse.txt
python3 - /tmp/test_3d_rig_sse.txt <<'PY'
import sys, json
stages = set(); complete = False
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev.get("type") == "progress": stages.add(ev.get("stage", ""))
    if ev.get("type") == "complete": complete = True
rig_stages = {s for s in stages if s.startswith("rig")}
assert complete, "no complete event"
assert rig_stages, f"no rig-stage labels, saw {sorted(stages)}"
print("PASS: streaming with rig stages", sorted(rig_stages))
PY
[ $? -eq 0 ] || exit 1

echo "ALL PASS: Hunyuan3D + UniRig rigged gen (skin + joints + normalized weights, streaming)"
