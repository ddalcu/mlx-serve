#!/usr/bin/env bash
# SDXL end-to-end over HTTP: discovery -> cold load -> generate -> unload.
#
# The numerics are pinned by the fixture parity tests in src/sdxl_*.zig (each
# component against diffusers, plus a full denoise against
# StableDiffusionXLPipeline). This script covers what those cannot see: that a
# checkpoint on disk is DISCOVERED, classified, routed to the image modality,
# cold-loaded by the gen dispatch, and answers the OpenAI image endpoint.
#
# Every one of those is a separate wiring point that fails silently in its own
# way — a repo that loads but is invisible to /v1/models, a model_type that
# routes nowhere, an engine arm that is never selected.
#
#   SDXL_MODEL=~/.mlx-serve/staging/sdxl-base-1.0 ./tests/test_sdxl_gen.sh [port]
#
# SKIPs cleanly when the checkpoint is absent.

set -uo pipefail

MODEL="${SDXL_MODEL:-$HOME/.mlx-serve/staging/sdxl-base-1.0}"
PORT="${1:-11398}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
# The image checks need numpy+PIL, which the system python often lacks. Prefer
# an interpreter that HAS them so section [4] runs instead of skipping — a
# skipped arm reads as a pass, and section [4] is the one that separates a real
# render from static.
PY_BIN="${SDXL_PY:-}"
if [ -z "$PY_BIN" ]; then
  for cand in "$HOME/.venvs/sdxl-oracle/bin/python" python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy, PIL" 2>/dev/null; then
      PY_BIN="$cand"; break
    fi
  done
fi
[ -n "$PY_BIN" ] || PY_BIN=python3
LOG="/tmp/sdxl_gen_test_$PORT.log"
ROOT="/tmp/sdxl_gen_root_$PORT"

PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

if [ ! -d "$MODEL" ] || [ ! -f "$MODEL/model_index.json" ]; then
  echo "SKIP: no SDXL checkpoint at $MODEL (set SDXL_MODEL)"; exit 0
fi
if [ ! -x "$BIN" ]; then
  echo "SKIP: $BIN not built (zig build -Doptimize=ReleaseFast)"; exit 0
fi

# A dedicated two-level root, so this exercises real discovery rather than
# --model, which would bypass the classification path entirely.
rm -f "/tmp/sdxl_img_$PORT.json" "/tmp/sdxl_img_$PORT.png" \
      "/tmp/sdxl_snap_$PORT.json" "/tmp/sdxl_chat_$PORT.json"
rm -rf "$ROOT"; mkdir -p "$ROOT/stabilityai"
ln -s "$MODEL" "$ROOT/stabilityai/sdxl-base-1.0"

cleanup() { kill %1 2>/dev/null; rm -rf "$ROOT"; }
trap cleanup EXIT

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
"$BIN" --serve --host 127.0.0.1 --port "$PORT" --model-dir "$ROOT" --log-level info > "$LOG" 2>&1 &
for _ in $(seq 1 40); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

echo "[1/7] discovery + classification"
MODELS=$(curl -s "http://127.0.0.1:$PORT/v1/models")
ID=$(echo "$MODELS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')")
check "the repo is discovered" "$ID" "stabilityai/sdxl-base-1.0"
ARCH=$(echo "$MODELS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['meta'].get('architecture','') if d.get('data') else '')")
# A diffusers repo carries NO root config.json — the arch is synthesized from
# model_index.json's declared pipeline class, on both the discovery and the
# routing side. A mismatch here is the class where one side sees a model and
# the other does not.
check "classified as sdxl" "$ARCH" "sdxl"
CAPS=$(echo "$MODELS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(d['data'][0].get('capabilities',[])) if d.get('data') else '')")
check "advertises the image capability" "$CAPS" "image"
# Symlinked checkpoints must be SIZED, not measured at zero — the .sym_link
# filter class.
BYTES=$(echo "$MODELS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0].get('bytes_on_disk',0) if d.get('data') else 0)")
if [ "$BYTES" -gt 1000000000 ]; then ok "sized on disk ($BYTES bytes)"; else bad "bytes_on_disk not measured ($BYTES)"; fi

echo "[2/7] a text request against an image model is refused, not prefilled"
CODE=$(curl -s -o /tmp/sdxl_chat_$PORT.json -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
check "chat on an image model 400s" "$CODE" "400"

echo "[3/7] generation (cold load on first request)"
CODE=$(curl -s -o /tmp/sdxl_img_$PORT.json -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"prompt\":\"a photo of a cat sitting on a wooden table\",\"size\":\"512x512\",\"steps\":8,\"seed\":42}")
if [ "$CODE" = "503" ] && grep -q "Insufficient memory for this media model" "$LOG"; then
  echo "  SKIP: the machine cannot fit SDXL right now (preflight refused; needs ~8.3 GB free)"
  echo "        This is the memory gate doing its job, not a wiring failure."
  echo "        Free memory and re-run, or pass --skip-mem-preflight to override."
  echo
  echo "SDXL: $PASS passed, $FAIL failed, generation SKIPPED (insufficient memory)"
  exit 0
fi
check "generation returns 200" "$CODE" "200"

python3 - "$PORT" <<'PY'
import base64, json, sys
port = sys.argv[1]
try:
    d = json.load(open(f"/tmp/sdxl_img_{port}.json"))
    b = base64.b64decode(d["data"][0]["b64_json"])
    open(f"/tmp/sdxl_img_{port}.png", "wb").write(b)
    print("  PASS: PNG magic" if b[:8] == b"\x89PNG\r\n\x1a\n" else "  FAIL: not a PNG")
    print(f"  PASS: {len(b)} bytes" if len(b) > 10000 else f"  FAIL: tiny payload {len(b)}")
except Exception as e:
    print(f"  FAIL: could not decode response: {e}")
PY

# The load line proves the ENGINE arm was selected — a 200 alone would also be
# returned by a different backend that happened to accept the request.
if grep -q "loaded unet: stages=" "$LOG"; then ok "the SDXL engine loaded (unet line in log)"; else bad "no SDXL unet load line"; fi
# The two tokenizers pad DIFFERENTLY, and padding both alike is invisible in
# the output. The boot lines are the only place that is observable.
if grep -q "loaded tokenizer_2: .*pad_id=0" "$LOG"; then ok "tokenizer_2 pads with 0"; else bad "tokenizer_2 pad_id not 0"; fi
if grep -q "loaded tokenizer: .*pad_id=49407" "$LOG"; then ok "tokenizer pads with EOS"; else bad "tokenizer pad_id not 49407"; fi

echo "[4/7] the render is an image, not static"
if "$PY_BIN" -c "import numpy, PIL" 2>/dev/null; then
  "$PY_BIN" - "$PORT" <<'PY'
import sys
import numpy as np
from PIL import Image
port = sys.argv[1]
a = np.asarray(Image.open(f"/tmp/sdxl_img_{port}.png").convert("RGB")).astype(float)
# Mean absolute difference between horizontally adjacent pixels. A real render
# lands 4-8; pure static lands 43-50. A parity test cannot catch a pipeline
# that renders noise if every component is individually right.
d = float(np.abs(np.diff(a, axis=1)).mean())
print(f"  {'PASS' if d < 20 else 'FAIL'}: adjacent-pixel diff {d:.1f} (static is 43-50)")
# A dead/black decode has near-zero variance and would pass the noise bar.
s = float(a.std())
print(f"  {'PASS' if s > 10 else 'FAIL'}: contrast std {s:.1f}")
PY
else
  echo "  SKIP: numpy/PIL unavailable"
fi

echo "[5/7] guidance-surface 400s, before any pixels are spent"
# `guidance`, `guidance_scale` and `timestep_spacing` are the three fields only
# a guided backend reads, and all three can refuse. Each refusal is asserted
# TWICE — once plain, once with "stream": true — because the streaming arm is
# where this class breaks: `sendError` on a socket that has already been handed
# `text/event-stream` headers writes a second status line into the event body,
# and curl reports the FIRST one, so a 400 emitted too late still reads as 200.
bad_request() { # <label> <json-fragment>
  local label="$1" frag="$2"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"prompt\":\"a red cube\",\"steps\":1,\"seed\":1,$frag}")
  check "$label" "$code" "400"
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"prompt\":\"a red cube\",\"steps\":1,\"seed\":1,\"stream\":true,$frag}")
  check "$label (streaming)" "$code" "400"
}
bad_request "an unserved timestep_spacing is refused" '"timestep_spacing":"quadratic"'
bad_request "guidance below the range is refused"     '"guidance":0.5'
bad_request "guidance above the range is refused"     '"guidance":99'
bad_request "guidance_scale is range-checked too"     '"guidance_scale":99'

# The same fields at legal values must not be refused — a range check that
# rejects everything would pass every assertion above.
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"prompt\":\"a red cube\",\"size\":\"512x512\",\"steps\":2,\"seed\":1,\"guidance\":7.5,\"timestep_spacing\":\"trailing\",\"negative_prompt\":\"blurry\"}")
check "a fully-specified guided request generates" "$CODE" "200"

echo "[6/7] size snapping and unload"
# 500 is not a multiple of 64; SDXL is trained on /64 buckets, so it snaps up
# rather than generating off-distribution.
CODE=$(curl -s -o /tmp/sdxl_snap_$PORT.json -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"prompt\":\"a red cube\",\"size\":\"500x500\",\"steps\":4,\"seed\":1}")
check "an unaligned size still generates" "$CODE" "200"
if "$PY_BIN" -c "import PIL" 2>/dev/null; then
  "$PY_BIN" - "$PORT" <<'PY'
import base64, io, json, sys
from PIL import Image
port = sys.argv[1]
d = json.load(open(f"/tmp/sdxl_snap_{port}.json"))
im = Image.open(io.BytesIO(base64.b64decode(d["data"][0]["b64_json"])))
print(f"  {'PASS' if im.size == (512, 512) else 'FAIL'}: 500 snapped to {im.size[0]} (want 512)")
PY
fi

# The route is `/v1/unload-model`. Asserted EXACTLY, and deliberately not
# "200 or 404": a typo'd path answers 404 and would have passed a tolerant
# check while proving nothing was unloaded.
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/unload-model" \
  -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}")
check "unload-model returns 200" "$CODE" "200"
STATE=$(curl -s "http://127.0.0.1:$PORT/v1/models" | "$PY_BIN" -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0].get('state','') if d.get('data') else '')")
check "the model is unloaded afterwards" "$STATE" "unloaded"

echo "[7/7] image-to-image: a high-contrast half-dark/half-bright source at LOW"
echo "      strength must retain its brightness split (the LTX I2V live-check pattern)"
SRC=/tmp/sdxl_img2img_src_$PORT.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 1024
rows = b""
for y in range(H):
    row = bytearray([0])
    for x in range(W):
        v = 235 if x >= W // 2 else 20
        row += bytes([v, v, v])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows))
       + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
PY
REQ=/tmp/sdxl_img2img_req_$PORT.json
python3 - "$SRC" "$REQ" <<'PY'
import sys, json, base64
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"prompt": "a photo", "size": "1024x1024", "steps": 8,
           "strength": 0.2, "image": b64, "seed": 7}, open(sys.argv[2], "w"))
PY
OUT2=/tmp/sdxl_img2img_out_$PORT.json
CODE=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d @"$REQ" -o "$OUT2" -w '%{http_code}')
if [ "$CODE" = "400" ] && grep -q "isn't available for this model" "$OUT2"; then
  echo "  SKIP: this checkpoint's VAE encoder didn't load (img2img unavailable) — a single-file checkpoint has no encoder.* tensors"
else
  check "img2img generation returns 200" "$CODE" "200"
  if grep -q "\[image\] img2img:" "$LOG"; then ok "img2img engagement log line"; else bad "no img2img engagement log line"; fi
  if "$PY_BIN" -c "import numpy" 2>/dev/null; then
    "$PY_BIN" - "$OUT2" <<'PY'
import sys, json, base64, zlib, struct
d = json.load(open(sys.argv[1]))
png = base64.b64decode(d["data"][0]["b64_json"])
assert png[:8] == bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
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
    if y % 32 == 0:
        for x in range(0, w // 2, 16): left_sum += line[x * 3]; n += 1
        for x in range(w // 2, w, 16): right_sum += line[x * 3]
left, right = left_sum / n, right_sum / n
print(f"  {'PASS' if right - left > 60 else 'FAIL'}: img2img strength=0.2 kept the split (left mean {left:.0f}, right mean {right:.0f})")
PY
  fi
fi

echo
echo "SDXL: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
