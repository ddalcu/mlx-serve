#!/usr/bin/env bash
# Pixal3D image-to-3D on the ONE main server: headless boot -> load the pack by absolute path -> /v1/models
# shows the "3d" capability -> one cutout (`image`) and two posed views (`images`, the default rig) each become
# a valid textured GLB -> the named 400s of the request surface -> streaming progress -> unload. Proves the
# TRELLIS.2-layout pack whose flows read "proj" routes to the Pixal3D engine, not TRELLIS.2.
#
# Skips gracefully when no pack is present. A pack is TencentARC/Pixal3D's layout (pipeline.json,
# pipeline_mv.json, ckpts/) plus dinov3/{config.json,model.safetensors} and naf/model.safetensors.
#
# Usage: PIXAL3D_MODEL=<pack dir> ./tests/test_pixal3d_3d.sh [port]
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11435}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

PACK="${PIXAL3D_MODEL:-$(ls -d ~/.mlx-serve/models/*/Pixal3D* 2>/dev/null | head -1)}"
[ -n "$PACK" ] || { echo "SKIP: no Pixal3D pack (set PIXAL3D_MODEL to its directory)"; exit 0; }
[ -f "$PACK/pipeline.json" ] || { echo "SKIP: $PACK has no pipeline.json"; exit 0; }

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

# 1. Load by absolute path -> ready with the "3d" capability, on the Pixal3D engine.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$PACK\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID' and x['state']=='ready' and '3d' in x.get('capabilities',[])]
assert m, 'Pixal3D not ready with 3d cap: '+json.dumps(d)
print('PASS: load-model by path -> Pixal3D ready, capabilities', m[0]['capabilities'])
" || { echo "FAIL: ready 3D model missing '3d' capability"; exit 1; }
grep -q "Pixal3D engine ready" "$LOG" || { echo "FAIL: the pack did not load on the Pixal3D engine"; exit 1; }
MV=0
grep -q "multi-view yes" "$LOG" && MV=1

# 2. Two cutouts: a red disc and a blue square with real alpha, transparent around them.
python3 - "$WORK" <<'PY'
import sys, struct, zlib
def png(path, inside, rgb):
    W = H = 512
    rows = b""
    for y in range(H):
        row = bytearray([0])
        for x in range(W):
            row += bytes(rgb + [255]) if inside(x - W // 2, y - H // 2) else bytes([0, 0, 0, 0])
        rows += bytes(row)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
png(sys.argv[1] + "/disc.png", lambda dx, dy: dx * dx + dy * dy < 170 ** 2, [200, 40, 40])
png(sys.argv[1] + "/square.png", lambda dx, dy: abs(dx) < 150 and abs(dy) < 150, [40, 60, 200])
PY
python3 - "$WORK" "$ID" <<'PY'
import json, base64, sys
work, mid = sys.argv[1], sys.argv[2]
a = base64.b64encode(open(work + "/disc.png", "rb").read()).decode()
b = base64.b64encode(open(work + "/square.png", "rb").read()).decode()
base = {"model": mid, "image": a, "steps": 4, "seed": 7, "decimation_target": 20000, "texture_size": 512}
mv = {k: v for k, v in base.items() if k != "image"}
mv["images"] = [a, b]
cam = [[1, 0, 0, 0], [0, 0, -1, -3], [0, 1, 0, 0], [0, 0, 0, 1]]
def dump(name, body): json.dump(body, open(f"{work}/{name}.json", "w"))
dump("req", dict(base, fov=30, roughness_scale=0.5, clearcoat=0.6))
dump("bad_rough", dict(base, roughness_scale=3))
dump("bad_coat", dict(base, clearcoat=-1))
dump("mv_stream", dict(mv, stream=True))
dump("bad_res", dict(base, resolution=512))
dump("bad_fov", dict(base, fov=0.4))
dump("bad_cam_single", dict(base, cameras=[cam]))
dump("bad_one_view", dict(mv, images=[a]))
dump("bad_both", dict(mv, image=a))
dump("bad_nine", dict(mv, images=[a] * 9))
dump("bad_five_rig", dict(mv, images=[a] * 5))
dump("bad_cam_count", dict(mv, cameras=[cam]))
PY

glb_ok() {
  python3 - "$1" <<'PY'
import sys, json, base64, struct
raw = open(sys.argv[1]).read()
if raw.startswith("data:") or "\ndata:" in raw:
    ev = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:")]
    r = [e for e in ev if e.get("type") == "complete"][0]
else:
    r = json.loads(raw)
assert r.get("format") == "glb", f"format={r.get('format')}"
b = base64.b64decode(r["data"])
magic, version, total = struct.unpack("<III", b[:12])
assert magic == 0x46546C67 and version == 2 and total == len(b), (hex(magic), version, total, len(b))
jlen, jtype = struct.unpack("<II", b[12:20])
g = json.loads(b[20:20 + jlen])
prim = g["meshes"][0]["primitives"][0]
pos = g["accessors"][prim["attributes"]["POSITION"]]
assert pos["count"] > 0, "empty mesh"
assert "TEXCOORD_0" in prim["attributes"], "no UVs"
assert len(g.get("images", [])) == 2, "want albedo + metallic-roughness images"
pbr = g["materials"][0]["pbrMetallicRoughness"]
assert "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr
print(f"{len(b)} byte textured GLB, {pos['count']} vertices")
PY
}

# 3. One view (few steps, small mesh and texture -> smoke, not quality) -> a valid textured GLB.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @"$WORK/req.json" -o "$WORK/resp.json" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: 3d gen http $code"; head -c 300 "$WORK/resp.json"; exit 1; }
out=$(glb_ok "$WORK/resp.json") || { echo "FAIL: GLB validity"; exit 1; }
grep -q "generating Pixal3D res=1024 views=1 cameras=default fov=30.0deg" "$LOG" || { echo "FAIL: no single-view Pixal3D engagement line"; exit 1; }
# The request's finish rides in the material: the roughness multiplier and the clearcoat extension.
python3 - "$WORK/resp.json" <<'PY' || { echo "FAIL: the finish did not reach the GLB material"; exit 1; }
import sys, json, base64, struct
b = base64.b64decode(json.load(open(sys.argv[1]))["data"])
g = json.loads(b[20:20 + struct.unpack("<I", b[12:16])[0]])
m = g["materials"][0]
assert abs(m["pbrMetallicRoughness"]["roughnessFactor"] - 0.5) < 1e-6, m
assert abs(m["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"] - 0.6) < 1e-6, m
PY
echo "PASS: one view -> $out"

# 4. Bad fields are 400s that name what is wrong; a missing image is a 400; the server survives.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after 3D gen"; exit 1; }
checks="bad_res:'resolution' bad_fov:'fov' bad_cam_single:'cameras' bad_one_view:'images' bad_both:not.both bad_nine:at.most bad_rough:'roughness_scale' bad_coat:'clearcoat'"
[ $MV = 1 ] && checks="$checks bad_five_rig:'cameras' bad_cam_count:'cameras'"
for f in $checks; do
  code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
    -d @"$WORK/${f%%:*}.json" -o "$WORK/err.json" -w "%{http_code}")
  [ "$code" = "400" ] && grep -q "${f##*:}" "$WORK/err.json" || { echo "FAIL: ${f%%:*} not a named 400 ($code)"; cat "$WORK/err.json"; exit 1; }
done
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: missing image returned $code (want 400)"; exit 1; }
echo "PASS: bad resolution / fov / cameras / view counts / missing image -> named 400s"

# 5. Two posed views on the default rig, streamed: stage progress + a complete event carrying the GLB.
if [ $MV = 1 ]; then
  code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
    -d @"$WORK/mv_stream.json" -o "$WORK/stream.txt" -w "%{http_code}")
  [ "$code" = "200" ] || { echo "FAIL: multi-view stream http $code"; exit 1; }
  grep -q 'pixal3d-shape' "$WORK/stream.txt" || { echo "FAIL: no pixal3d stage progress in the stream"; exit 1; }
  out=$(glb_ok "$WORK/stream.txt") || { echo "FAIL: multi-view GLB validity"; exit 1; }
  grep -q "generating Pixal3D res=1024 views=2 cameras=default fov=20.0deg" "$LOG" || { echo "FAIL: no multi-view engagement line"; exit 1; }
  echo "PASS: two views, streamed -> stage progress + $out"
else
  echo "SKIP: the pack has no pipeline_mv.json (multi-view)"
fi

# 6. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID']
assert m and m[0]['state']=='unloaded', 'Pixal3D should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> Pixal3D unloaded (stub retained)')
"

echo "ALL PASS: Pixal3D 3D gen (headless boot, one view + two views, textured GLB validity, field validation, streaming)"
