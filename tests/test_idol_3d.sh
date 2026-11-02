#!/usr/bin/env bash
# Idol image-to-3D-Gaussians on the ONE main server: headless boot -> load the pack by absolute path -> /v1/models
# shows the "3d" capability -> a photo with a background (BiRefNet cuts it) and a cutout with real alpha (it does
# not) each become a GLB carrying KHR_gaussian_splatting, streamed with stage progress -> the named 400s of the
# request surface -> unload.
#
# Skips gracefully when no pack is present. A pack is `tests/convert_idol_weights.py`'s output (config.json,
# denoiser|decoder|decoder_gs|birefnet.safetensors, dinov3/, vae/) or speridlabs/idol's own folder plus those two.
#
# Usage: IDOL_MODEL=<pack dir> ./tests/test_idol_3d.sh [port]
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11436}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

PACK="${IDOL_MODEL:-$(ls -d ~/.mlx-serve/models/local/Idol* ~/.mlx-serve/models/*/Idol* 2>/dev/null | head -1)}"
[ -n "$PACK" ] || { echo "SKIP: no Idol pack (set IDOL_MODEL to its directory)"; exit 0; }
[ -f "$PACK/vae/model.safetensors" ] || { echo "SKIP: $PACK is not a complete Idol pack"; exit 0; }

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

# 1. Load by absolute path -> ready with the "3d" capability, on the Idol plugin.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$PACK\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID' and x['state']=='ready' and '3d' in x.get('capabilities',[])]
assert m, 'Idol not ready with 3d cap: '+json.dumps(d)
assert m[0]['meta']['architecture']=='idol', m[0]['meta']
print('PASS: load-model by path -> Idol ready, capabilities', m[0]['capabilities'])
" || { echo "FAIL: ready 3D model missing '3d' capability"; exit 1; }
grep -q "idol ready through its plugin table" "$LOG" || { echo "FAIL: the pack did not load through the Idol plugin"; exit 1; }

# 2. A photo-like image (a coloured ellipse on a bright gradient, no alpha) and a cutout of the same ellipse.
python3 - "$WORK" <<'PY'
import sys, struct, zlib
W = H = 256
def png(path, pixel, mode):
    rows = b""
    for y in range(H):
        row = bytearray([0])
        for x in range(W):
            row += bytes(pixel(x, y))
        rows += bytes(row)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, mode, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
def inside(x, y):
    return ((x - 128) / 70) ** 2 + ((y - 128) / 90) ** 2 < 1
def photo(x, y):
    if inside(x, y):
        return [190 + (x - 60) // 4, 60, 40 + (y - 40) // 3]
    return [225 - y // 5, 235 - x // 6, 245]
def cutout(x, y):
    return photo(x, y) + [255] if inside(x, y) else [0, 0, 0, 0]
png(sys.argv[1] + "/photo.png", photo, 2)
png(sys.argv[1] + "/cutout.png", cutout, 6)
PY
python3 - "$WORK" "$ID" <<'PY'
import json, base64, sys
work, mid = sys.argv[1], sys.argv[2]
p = base64.b64encode(open(work + "/photo.png", "rb").read()).decode()
c = base64.b64encode(open(work + "/cutout.png", "rb").read()).decode()
base = {"model": mid, "steps": 6, "seed": 7, "num_gaussians": 32768}
def dump(name, body): json.dump(body, open(f"{work}/{name}.json", "w"))
dump("photo", dict(base, image=p, mode="real", stream=True, geometry="splat"))
dump("cutout", dict(base, image=c, guidance_scale=2.0, stream=True, geometry="splat"))
dump("discs", dict(base, image=c, stream=True))
dump("bad_geometry", dict(base, image=c, geometry="mesh"))
dump("bad_steps", dict(base, image=c, steps=0))
dump("bad_gaussians", dict(base, image=c, num_gaussians=100))
dump("bad_mode", dict(base, image=c, mode="cartoon"))
dump("bad_guidance", dict(base, image=c, guidance_scale=99))
dump("bad_images", dict(base, images=[c, c]))
PY

splat_ok() {  # <sse file> <expected count>
  python3 - "$1" "$2" <<'PY'
import sys, json, base64, struct
ev = [json.loads(l[5:]) for l in open(sys.argv[1]).read().splitlines() if l.startswith("data:")]
r = [e for e in ev if e.get("type") == "complete"][0]
assert r.get("format") == "glb", f"format={r.get('format')}"
b = base64.b64decode(r["data"])
magic, version, total = struct.unpack("<III", b[:12])
assert magic == 0x46546C67 and version == 2 and total == len(b), (hex(magic), version, total, len(b))
jlen, _ = struct.unpack("<II", b[12:20])
g = json.loads(b[20:20 + jlen])
prim = g["meshes"][0]["primitives"][0]
assert prim["mode"] == 0 and "KHR_gaussian_splatting" in prim["extensions"], prim
a = g["accessors"][prim["attributes"]["POSITION"]]
assert a["count"] == int(sys.argv[2]), a["count"]
assert max(abs(v) for v in a["min"] + a["max"]) < 1.5, (a["min"], a["max"])
for k in ("KHR_gaussian_splatting:ROTATION", "KHR_gaussian_splatting:SCALE", "KHR_gaussian_splatting:OPACITY", "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", "COLOR_0"):
    assert k in prim["attributes"], k
stages = sorted({e["stage"] for e in ev if e.get("type") == "progress"})
print(f"{len(b)} byte splat GLB, {a['count']} Gaussians, stages {stages}")
PY
}

# 3. The photo: BiRefNet cuts it, then the whole pipeline streams to a splat GLB.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @"$WORK/photo.json" -o "$WORK/photo.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: photo gen http $code"; head -c 300 "$WORK/photo.txt"; exit 1; }
out=$(splat_ok "$WORK/photo.txt" 32768) || { echo "FAIL: photo splat validity"; exit 1; }
grep -q '"stage":"idol-matte"' "$WORK/photo.txt" || { echo "FAIL: a photo with no alpha skipped the background matte"; exit 1; }
grep -q "generating Idol steps=6 gaussians=32768" "$LOG" || { echo "FAIL: no Idol engagement line"; exit 1; }
echo "PASS: photo -> $out"

# 4. A cutout with real alpha is used as it is: no matte stage.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @"$WORK/cutout.json" -o "$WORK/cutout.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: cutout gen http $code"; head -c 300 "$WORK/cutout.txt"; exit 1; }
out=$(splat_ok "$WORK/cutout.txt" 32768) || { echo "FAIL: cutout splat validity"; exit 1; }
grep -q '"stage":"idol-matte"' "$WORK/cutout.txt" && { echo "FAIL: a cutout with real alpha was matted again"; exit 1; }
echo "PASS: cutout -> $out (no matte)"

# 4b. The default geometry is the disc mesh: triangles with one baked atlas texel per Gaussian, no extension.
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d @"$WORK/discs.json" -o "$WORK/discs.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: discs gen http $code"; exit 1; }
python3 - "$WORK/discs.txt" <<'PY' || { echo "FAIL: disc mesh validity"; exit 1; }
import sys, json, base64, struct
ev = [json.loads(l[5:]) for l in open(sys.argv[1]).read().splitlines() if l.startswith("data:")]
b = base64.b64decode([e for e in ev if e.get("type") == "complete"][0]["data"])
jlen, _ = struct.unpack("<II", b[12:20]); g = json.loads(b[20:20 + jlen])
prim = g["meshes"][0]["primitives"][0]
assert prim["mode"] == 4 and "extensions" not in prim and "TEXCOORD_0" in prim["attributes"], prim
assert len(g["images"]) == 1 and g["accessors"][prim["attributes"]["POSITION"]]["count"] % 4 == 0
print(f"{len(b)} byte disc mesh, {g['accessors'][prim['indices']]['count'] // 6} discs")
PY
echo "PASS: default geometry -> a textured disc mesh"

# 5. Bad fields are 400s that name what is wrong; a missing image is a 400; the server survives.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after 3D gen"; exit 1; }
for f in bad_steps:"'steps'" bad_gaussians:"'num_gaussians'" bad_mode:"'mode'" bad_guidance:"'guidance_scale'" bad_images:"Pixal3D" bad_geometry:"'geometry'"; do
  code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
    -d @"$WORK/${f%%:*}.json" -o "$WORK/err.json" -w "%{http_code}")
  [ "$code" = "400" ] && grep -q "${f##*:}" "$WORK/err.json" || { echo "FAIL: ${f%%:*} not a named 400 ($code)"; cat "$WORK/err.json"; exit 1; }
done
code=$(api /v1/3d/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: missing image returned $code (want 400)"; exit 1; }
echo "PASS: bad steps / Gaussians / mode / guidance / several views / missing image -> named 400s"

# 6. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID']
assert m and m[0]['state']=='unloaded', 'Idol should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> Idol unloaded (stub retained)')
"

echo "ALL PASS: Idol 3D gen (headless boot, photo + cutout, splat GLB validity, field validation, streaming)"
