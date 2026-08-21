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

echo "[1/6] discovery + classification"
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

echo "[2/6] a text request against an image model is refused, not prefilled"
CODE=$(curl -s -o /tmp/sdxl_chat_$PORT.json -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
check "chat on an image model 400s" "$CODE" "400"

echo "[3/6] generation (cold load on first request)"
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

echo "[4/6] the render is an image, not static"
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

echo "[5/6] guidance-surface 400s, before any pixels are spent"
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

echo "[6/6] size snapping and unload"
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

echo
echo "SDXL: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
