#!/usr/bin/env bash
# Krea-2-Turbo image gen on the ONE main server: headless boot -> load the Krea
# model by absolute path -> generate (at a NON-1024 size, exercising the relaxed
# Krea size gate) -> assert a valid PNG of the requested size -> coexist with a
# chat model -> unload. Proves the image-backend seam routes `krea*` to the Krea
# engine and that chat + Krea coexist on one process/port.
#
# Skips gracefully when no Krea model is present. The Krea dir must be assembled
# (transformer_*.safetensors + text_encoder/ + vae/ + tokenizer/ + config.json
# with {"model_type":"krea2_turbo"}).
#
# Usage: KREA_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_krea_gen.sh [port]
set -uo pipefail
PORT="${1:-11399}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

KREA="${KREA_MODEL:-$(ls -d ~/.mlx-serve/models/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$KREA" ] || { echo "SKIP: no Krea model (set KREA_MODEL to an assembled dir)"; exit 0; }
[ -f "$KREA/config.json" ] || { echo "SKIP: $KREA has no config.json (assemble the dir + add {\"model_type\":\"krea2_turbo\"})"; exit 0; }

# Headless: --model-dir anywhere; the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_krea_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_krea_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 1200 "http://127.0.0.1:$PORT$1" "${@:2}"; }
KREA_ID="$(basename "$KREA")"

# 1. Headless: no default model.
N=$(api /v1/models | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"]))')
[ "$N" = "0" ] || { echo "FAIL: headless /v1/models should be empty, got $N"; exit 1; }
echo "PASS: headless boot, /v1/models empty"

# 2. Load Krea by absolute path -> ready with "image" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$KREA\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$KREA_ID' and x['state']=='ready' and 'image' in x.get('capabilities',[])]
assert m, 'Krea not ready with image cap: '+json.dumps(d)
print('PASS: load-model by path -> Krea ready, capabilities', m[0]['capabilities'])
"

# 3. Generate at 512x512 (NON-1024 -> exercises the relaxed Krea size gate) -> PNG of that size.
api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$KREA_ID\",\"prompt\":\"a red fox in the snow\",\"size\":\"512x512\",\"steps\":8}" \
  -o /tmp/test_krea_img.json -w ''
python3 - /tmp/test_krea_img.json <<'PY'
import sys,json,base64,struct
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
assert (w,h)==(512,512), f"expected 512x512, got {w}x{h}"
print(f"PASS: /v1/images/generations (Krea, relaxed size) -> {len(b)} byte PNG {w}x{h}")
PY

# 4. Server survives the gen.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after Krea gen"; exit 1; }

# 4b. image-to-image: half-dark/half-bright source at low strength keeps its
# split; wrong cond_weights count is a 400 (Krea taps 12 encoder layers).
SRC=/tmp/test_krea_img2img_src.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 512
rows = b""
for y in range(H):
    row = bytearray([0])
    for x in range(W):
        v = 235 if x >= W // 2 else 20
        row += bytes([v, v, v])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY
python3 - "$SRC" /tmp/test_krea_i2i_req.json <<PY
import json, base64, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"model": "$KREA_ID", "prompt": "a photo", "size": "512x512", "steps": 8,
           "strength": 0.25, "image": b64, "seed": 7,
           "cond_weights": "1 1 1 1 1 1 1 1 1 1 1 1"}, open(sys.argv[2], "w"))
PY
code=$(api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_krea_i2i_req.json -o /tmp/test_krea_i2i.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: krea img2img http $code"; head -c 300 /tmp/test_krea_i2i.json; exit 1; }
grep -q "\[image\] img2img:" /tmp/test_krea_server.log || { echo "FAIL: no img2img engagement log"; exit 1; }
python3 - /tmp/test_krea_i2i.json <<'PY'
import sys, json, base64, zlib, struct
png = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
pos, idat, w, h = 8, b"", 0, 0
while pos < len(png):
    ln, typ = struct.unpack(">I4s", png[pos:pos+8]); data = png[pos+8:pos+8+ln]; pos += 12 + ln
    if typ == b"IHDR": w, h, _, ct = struct.unpack(">IIBB", data[:10]); assert ct == 2
    elif typ == b"IDAT": idat += data
raw = zlib.decompress(idat)
stride = w * 3
prev = bytearray(stride)
left_sum = right_sum = 0; n = 0
for y in range(h):
    f = raw[y * (stride + 1)]
    line = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
    for i in range(stride):
        a = line[i - 3] if i >= 3 else 0
        b = prev[i]
        c = prev[i - 3] if i >= 3 else 0
        if f == 1: line[i] = (line[i] + a) & 0xFF
        elif f == 2: line[i] = (line[i] + b) & 0xFF
        elif f == 3: line[i] = (line[i] + (a + b) // 2) & 0xFF
        elif f == 4:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            line[i] = (line[i] + pr) & 0xFF
    prev = line
    if y % 16 == 0:
        for x in range(0, w // 2, 8): left_sum += line[x * 3]; n += 1
        for x in range(w // 2, w, 8): right_sum += line[x * 3]
left, right = left_sum / n, right_sum / n
print(f"PASS: krea img2img strength=0.25 kept the split (left {left:.0f}, right {right:.0f})")
assert right - left > 60, f"img2img lost the source structure (left {left:.0f}, right {right:.0f})"
PY
[ $? -eq 0 ] || exit 1
code=$(api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$KREA_ID\",\"prompt\":\"x\",\"steps\":2,\"cond_weights\":\"1 1 1\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: 3 cond_weights on Krea returned $code (want 400)"; exit 1; }
echo "PASS: 3 cond_weights on Krea -> 400 (needs 12)"

# Instruction editing is FLUX.2-only (Krea has no edit training) → 400.
code=$(api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$KREA_ID\",\"prompt\":\"x\",\"mode\":\"edit\",\"image\":\"aGk=\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: edit mode on Krea returned $code (want 400)"; exit 1; }
echo "PASS: mode=edit on Krea -> 400 (FLUX.2-only)"

# 5. Coexistence with a chat model.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
ready={x['id'] for x in d if x['state']=='ready'}
assert '$KREA_ID' in ready and '$CHAT_ID' in ready, 'both should be resident: '+json.dumps([(x['id'],x['state']) for x in d])
print('PASS: chat + Krea RESIDENT together:', sorted(ready))
"
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while Krea resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with Krea also resident"
fi

# 6. Unload Krea -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$KREA_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$KREA_ID']
assert m and m[0]['state']=='unloaded', 'Krea should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> Krea unloaded (stub retained)')
"

echo "ALL PASS: Krea-2-Turbo image gen (headless boot, load->gen->unload, relaxed size, coexistence)"
