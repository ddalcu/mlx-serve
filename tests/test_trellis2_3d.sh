#!/usr/bin/env bash
# TRELLIS.2 image-to-3D on the ONE main server: headless boot -> load the pack by absolute path ->
# /v1/models shows the "3d" capability -> POST /v1/3d/generations with a cutout -> assert a valid
# textured GLB (magic, JSON chunk, positions + UVs, albedo + metallic-roughness images) -> field
# validation 400s -> streaming progress -> unload. Proves the pipeline.json-only pack routes to
# the .mesh slot and the whole GPU + CPU pipeline runs behind the endpoint.
#
# Skips gracefully when no pack is present. A pack is microsoft/TRELLIS.2-4B's layout: pipeline.json,
# ckpts/, microsoft/TRELLIS-image-large/ckpts/ss_dec_*, dinov3/{config.json,model.safetensors}.
#
# Usage: TRELLIS2_MODEL=<pack dir> [TRELLIS2_RES=1024] ./tests/test_trellis2_3d.sh [port]
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11433}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

PACK="${TRELLIS2_MODEL:-$(ls -d ~/.mlx-serve/models/*/TRELLIS.2* 2>/dev/null | head -1)}"
[ -n "$PACK" ] || { echo "SKIP: no TRELLIS.2 pack (set TRELLIS2_MODEL to its directory)"; exit 0; }
[ -f "$PACK/pipeline.json" ] || { echo "SKIP: $PACK has no pipeline.json"; exit 0; }
RES="${TRELLIS2_RES:-1024}"

WORK="$(mktemp -d)"
LOG="$WORK/server.log"
mkdir -p "$WORK/home" "$WORK/empty"
# Scratch HOME: a saved model-settings profile or providers.json must not shape this server.
HOME="$WORK/home" "$BIN" --serve --model-dir "$WORK/empty" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$WORK"' EXIT
srv_wait_health "http://127.0.0.1:$PORT" 60 1 "$SRV" || kill -0 "$SRV" 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 "$LOG"; exit 1; }

api() { curl -s -m 3600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
ID="$(basename "$PACK")"

# 1. Load by absolute path -> ready with the "3d" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$PACK\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID' and x['state']=='ready' and '3d' in x.get('capabilities',[])]
assert m, 'TRELLIS.2 not ready with 3d cap: '+json.dumps(d)
print('PASS: load-model by path -> TRELLIS.2 ready, capabilities', m[0]['capabilities'])
" || { echo "FAIL: ready 3D model missing '3d' capability"; exit 1; }

# 2. A cutout: a red disc with real alpha (the subject), transparent around it.
python3 - "$WORK/src.png" <<'PY'
import sys, struct, zlib
W = H = 512
rows = b""
for y in range(H):
    row = bytearray([0])
    for x in range(W):
        d2 = (x - W // 2) ** 2 + (y - H // 2) ** 2
        row += bytes([200, 40, 40, 255]) if d2 < (W // 3) ** 2 else bytes([0, 0, 0, 0])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY
python3 - "$WORK/src.png" "$WORK" "$ID" "$RES" <<'PY'
import json, base64, sys
src, work, mid, res = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
b64 = base64.b64encode(open(src, "rb").read()).decode()
base = {"model": mid, "image": b64, "resolution": res, "steps": 4, "seed": 7,
        "decimation_target": 20000, "texture_size": 512, "roughness_scale": 0.4, "clearcoat": 1.0}
json.dump(base, open(work + "/req.json", "w"))
json.dump(dict(base, stream=True), open(work + "/stream_req.json", "w"))
json.dump(dict(base, resolution=768), open(work + "/bad_res.json", "w"))
json.dump(dict(base, texture_size=1000), open(work + "/bad_tex.json", "w"))
json.dump(dict(base, roughness_scale=0), open(work + "/bad_rough.json", "w"))
json.dump(dict(base, clearcoat=2), open(work + "/bad_coat.json", "w"))
json.dump({k: v for k, v in dict(base, steps=0).items()}, open(work + "/bad_steps.json", "w"))
mv = {k: v for k, v in base.items() if k != "image"}
json.dump(dict(mv, images=[b64, b64]), open(work + "/bad_images.json", "w"))
PY

# 3. Generate (few steps, small mesh and texture -> smoke, not quality) -> a valid textured GLB.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @"$WORK/req.json" -o "$WORK/resp.json" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: 3d gen http $code"; head -c 300 "$WORK/resp.json"; exit 1; }
python3 - "$WORK/resp.json" <<'PY'
import sys, json, base64, struct
r = json.load(open(sys.argv[1]))
assert r.get("format") == "glb", f"format={r.get('format')}"
b = base64.b64decode(r["data"])
magic, version, total = struct.unpack("<III", b[:12])
assert magic == 0x46546C67, hex(magic)
assert version == 2 and total == len(b), (version, total, len(b))
jlen, jtype = struct.unpack("<II", b[12:20])
assert jtype == 0x4E4F534A
g = json.loads(b[20:20 + jlen])
prim = g["meshes"][0]["primitives"][0]
pos = g["accessors"][prim["attributes"]["POSITION"]]
assert pos["count"] > 0, "empty mesh"
assert "TEXCOORD_0" in prim["attributes"], "no UVs"
assert len(g.get("images", [])) == 2, "want albedo + metallic-roughness images"
pbr = g["materials"][0]["pbrMetallicRoughness"]
assert "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr
# The request's finish rides in the material: the roughness multiplier and the clearcoat extension.
assert abs(pbr["roughnessFactor"] - 0.4) < 1e-6, pbr
coat = g["materials"][0]["extensions"]["KHR_materials_clearcoat"]
assert coat["clearcoatFactor"] == 1.0 and "KHR_materials_clearcoat" in g["extensionsUsed"], coat
print(f"PASS: /v1/3d/generations -> {len(b)} byte textured GLB, {pos['count']} vertices")
PY
[ $? -eq 0 ] || { echo "FAIL: GLB validity"; exit 1; }

# 4. Server survives; bad fields are 400s that name the field; a missing image is a 400.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after 3D gen"; exit 1; }
for f in bad_res:resolution bad_tex:texture_size bad_steps:steps bad_images:Pixal3D bad_rough:roughness_scale bad_coat:clearcoat; do
  code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
    -d @"$WORK/${f%%:*}.json" -o "$WORK/err.json" -w "%{http_code}")
  [ "$code" = "400" ] && grep -q "${f##*:}" "$WORK/err.json" || { echo "FAIL: ${f##*:} not a named 400 ($code)"; cat "$WORK/err.json"; exit 1; }
done
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: missing image returned $code (want 400)"; exit 1; }
echo "PASS: bad resolution / texture_size / steps / 'images' (one view only) / roughness_scale / clearcoat / missing image -> named 400s"

# 5. Streaming: stage progress events + a complete event carrying the GLB.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @"$WORK/stream_req.json" -o "$WORK/stream.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream 3d gen http $code"; exit 1; }
grep -q 'trellis-shape' "$WORK/stream.txt" || { echo "FAIL: no trellis stage progress in the stream"; exit 1; }
grep -q '"type":"complete"' "$WORK/stream.txt" || { echo "FAIL: no complete event in stream"; exit 1; }
echo "PASS: streaming -> stage progress + SSE complete event"

# 6. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID']
assert m and m[0]['state']=='unloaded', 'TRELLIS.2 should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> TRELLIS.2 unloaded (stub retained)')
"

echo "ALL PASS: TRELLIS.2 3D gen (headless boot, load->gen->unload, textured GLB validity, field validation, streaming)"
