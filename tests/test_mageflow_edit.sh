#!/usr/bin/env bash
# Mage-Flow-Edit: the capabilities the app's UI now advertises for this backend,
# and the ones it deliberately hides. `test_image_gen.sh` is FLUX-shaped (img2img,
# LoRA, conditioning rebalance) and Mage-Flow has none of those, so the checks
# that matter here live in their own script.
#
# Skips gracefully when the checkpoint isn't downloaded (~17 GB).
# Usage: MAGEFLOW_EDIT_MODEL=<dir> ./tests/test_mageflow_edit.sh [port]
set -uo pipefail
PORT="${1:-11402}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${MAGEFLOW_EDIT_MODEL:-$HOME/.mlx-serve/models/microsoft/Mage-Flow-Edit-Turbo}"
[ -f "$MODEL/model_index.json" ] || { echo "SKIP: no Mage-Flow-Edit checkpoint at $MODEL"; exit 0; }

LOG=/tmp/test_mageflow_edit_server.log
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done

# The edit capability is gated on the checkpoint (dir name — both repos ship
# byte-identical configs), and the banner is the only place it's observable.
grep -q "EDIT (multi-reference in-context editor)" "$LOG" \
  || { echo "FAIL: model did not come up in EDIT mode"; grep MageFlow "$LOG" | tail -2; exit 1; }
echo "PASS: checkpoint detected as EDIT-capable"

# A deliberately NON-square source, so "keeps the source resolution" and "keeps
# the source aspect" are distinguishable from "returned the default 1024²".
SRC=/tmp/test_mageflow_src.png
python3 - "$SRC" <<'PY'
import struct, sys, zlib
W, H = 1152, 768
rows = b"".join(b"\0" + bytes(v for x in range(W) for v in (x * 255 // W, y * 255 // H, 96)) for y in range(H))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))
PY
SRC_DIMS=$(python3 -c 'import struct;d=open("/tmp/test_mageflow_src.png","rb").read();print("%dx%d"%struct.unpack(">II",d[16:24]))')

post() { # body-json -> "WxH" (or "ERR <code>"). Body and status go to separate
         # sinks: BSD head has no `-n -1`, so splitting a combined stream isn't
         # portable here.
  local body="$1" tmp code
  tmp=$(mktemp)
  code=$(curl -s -o "$tmp" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
         -H 'Content-Type: application/json' -d @- <<<"$body")
  if [ "$code" != "200" ]; then echo "ERR $code"; rm -f "$tmp"; return; fi
  python3 -c 'import base64,json,struct,sys
d=json.load(open(sys.argv[1])); png=base64.b64decode(d["data"][0]["b64_json"])
assert png[:8]==b"\x89PNG\r\n\x1a\n"; print("%dx%d"%struct.unpack(">II",png[16:24]))' "$tmp"
  rm -f "$tmp"
}
b64() { python3 -c 'import base64,sys;print(base64.b64encode(open(sys.argv[1],"rb").read()).decode())' "$1"; }
SRC_B64=$(b64 "$SRC")

# ── text-to-image still works on an edit checkpoint (same DiT) ──
d=$(post "{\"prompt\":\"a red apple\",\"size\":\"512x512\",\"steps\":4,\"seed\":1}")
[ "$d" = "512x512" ] || { echo "FAIL: txt2img returned $d (want 512x512)"; exit 1; }
echo "PASS: text-to-image on the edit checkpoint -> $d"

# ── no size = the source's own resolution (max_size = source size) ──
d=$(post "{\"prompt\":\"make it winter\",\"mode\":\"edit\",\"image\":\"$SRC_B64\",\"steps\":4,\"seed\":3}")
[ "$d" = "$SRC_DIMS" ] || { echo "FAIL: sizeless edit returned $d (want the source's $SRC_DIMS)"; exit 1; }
grep -q "size matched to source" "$LOG" || { echo "FAIL: no match-to-source log line"; exit 1; }
echo "PASS: edit without 'size' keeps the source resolution ($d)"

# ── a source ABOVE the backend's 2048 dimension cap keeps its aspect ──
# Every fixture above is under the cap, so they cannot see the failure this
# guards: `fitAspect` preserved the aspect and `normalizeSize` then clamped each
# dimension INDEPENDENTLY, squaring a 4032x3024 phone photo off to 2048x2048.
# Any real photo lands here, via the plain `client.images.edit(image=…)` call.
BIG=/tmp/test_mageflow_big.png
python3 - "$BIG" <<'PY'
import struct, sys, zlib
W, H = 3024, 2016  # 3:2, long edge well over the 2048 cap
row = b"\0" + bytes([200, 120, 60] * W)   # content is irrelevant; only dims are
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(row * H, 1)) + chunk(b"IEND", b""))
PY
BIG_B64=$(b64 "$BIG")
d=$(post "{\"prompt\":\"make it winter\",\"mode\":\"edit\",\"image\":\"$BIG_B64\",\"steps\":4,\"seed\":7}")
case "$d" in ERR*) echo "FAIL: oversized-source edit -> $d"; exit 1;; esac
python3 - "$d" <<'PY'
import sys
ow, oh = (int(v) for v in sys.argv[1].split("x"))
assert max(ow, oh) <= 2048, f"exceeded the backend cap: {ow}x{oh}"
assert abs(ow / oh - 3024 / 2016) < 0.05, f"aspect squashed: {ow}x{oh} from a 3:2 source"
PY
[ $? -eq 0 ] || exit 1
echo "PASS: source above the cap scales instead of squashing ($d)"

# ── an explicit size is a BUDGET; the source's aspect ratio still wins ──
d=$(post "{\"prompt\":\"make it winter\",\"mode\":\"edit\",\"image\":\"$SRC_B64\",\"size\":\"1024x1024\",\"steps\":4,\"seed\":3}")
[ "$d" != "1024x1024" ] || { echo "FAIL: square request squashed a 3:2 reference"; exit 1; }
python3 - "$d" "$SRC_DIMS" <<'PY'
import sys
ow, oh = (int(v) for v in sys.argv[1].split("x")); sw, sh = (int(v) for v in sys.argv[2].split("x"))
assert abs(ow / oh - sw / sh) < 0.05, f"aspect drifted: {ow}x{oh} vs source {sw}x{sh}"
assert abs(ow * oh - 1024 * 1024) / (1024 * 1024) < 0.15, f"budget ignored: {ow*oh} px"
PY
[ $? -eq 0 ] || exit 1
echo "PASS: explicit size = pixel budget at the source's aspect ($d)"

# ── multi-reference composition engages the second reference ──
d=$(post "{\"prompt\":\"put the object from image 2 into image 1\",\"mode\":\"edit\",\"image\":\"$SRC_B64\",\"ref_images\":[\"$SRC_B64\"],\"steps\":4,\"seed\":5}")
case "$d" in ERR*) echo "FAIL: multi-reference edit -> $d"; exit 1;; esac
grep -q "edit ref 2" "$LOG" || { echo "FAIL: second reference never reached the engine"; exit 1; }
echo "PASS: multi-reference composition -> $d"

# ── capabilities the app now HIDES must still be honest 400s on the wire ──
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
  -H 'Content-Type: application/json' -d "{\"prompt\":\"x\",\"image\":\"$SRC_B64\",\"strength\":0.5}")
[ "$code" = "400" ] || { echo "FAIL: variation on a no-img2img backend returned $code (want 400)"; exit 1; }
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
  -H 'Content-Type: application/json' -d '{"prompt":"x","lora_path":"/tmp/nope.safetensors"}')
[ "$code" = "400" ] || { echo "FAIL: LoRA on a no-LoRA backend returned $code (want 400)"; exit 1; }
echo "PASS: variation + LoRA -> 400 (the UI hides both for this model)"

# ── OpenAI multipart surface reaches the same engine ──
d=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/edits" \
  -F "prompt=make it winter" -F "image=@$SRC;type=image/png" \
  | python3 -c 'import base64,json,struct,sys
d=json.load(sys.stdin); png=base64.b64decode(d["data"][0]["b64_json"]); print("%dx%d"%struct.unpack(">II",png[16:24]))')
[ "$d" = "$SRC_DIMS" ] || { echo "FAIL: /v1/images/edits returned $d (want $SRC_DIMS)"; exit 1; }
echo "PASS: OpenAI /v1/images/edits (no size) -> source resolution ($d)"

# ── the multipart 'model' FIELD must select the model ──
# Model resolution runs BEFORE the route translates the form to JSON, so a
# JSON-only scan found no `"model":` key and every edit silently got
# default-model semantics. Live via Open WebUI (2026-07-25): an edit naming a
# Mage-Flow model ran against the default CHAT model and 400'd "does not support
# this media modality"; headless with no default it 503'd "No default model
# configured". Everything above boots with `--model $MODEL`, so the default is
# ALREADY the model under test and none of it can see this class — this section
# needs a server with NO default, where the id has to come from the form.
MODEL_ROOT="$(cd "$MODEL/../.." && pwd)"
MODEL_ID="$(basename "$(dirname "$MODEL")")/$(basename "$MODEL")"
HPORT=$((PORT + 40))
"$BIN" --serve --port "$HPORT" --model-dir "$MODEL_ROOT" >/tmp/test_mageflow_edit_headless.log 2>&1 &
HSRV=$!
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$HPORT/health" >/dev/null 2>&1 && break
  kill -0 $HSRV 2>/dev/null || { echo "FAIL: headless server did not start"; exit 1; }
  sleep 1
done
hcode=$(curl -s -o /tmp/test_mageflow_edit_headless_body.json -w '%{http_code}' --max-time 900 \
  -X POST "http://127.0.0.1:$HPORT/v1/images/edits" \
  -F "model=$MODEL_ID" -F "prompt=make it winter" -F "image=@$SRC;type=image/png")
kill $HSRV 2>/dev/null
[ "$hcode" = "200" ] || {
  echo "FAIL: edit naming '$MODEL_ID' on a server with no default returned $hcode (want 200)"
  echo "      503 = the form's 'model' field was ignored and it fell back to the default"
  head -c 300 /tmp/test_mageflow_edit_headless_body.json; echo
  exit 1
}
echo "PASS: /v1/images/edits loads the model named in the form (no default configured)"

echo "ALL PASS: Mage-Flow-Edit (detection, match-source, aspect budget, multi-ref, honest 400s, OpenAI surface, form-named model)"
