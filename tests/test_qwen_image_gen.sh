#!/usr/bin/env bash
# Qwen-Image-2.1 on the ONE main server: headless boot -> load the converted
# pack by absolute path -> txt2img PNG of the requested size -> real CFG engages
# (two forwards per step, by the log) and changes the render -> img2img ->
# instruction edit (mode:"edit", the pack's Qwen3-VL tower): happy path,
# no-size output aspect from the LAST reference, multi-reference + the cap,
# edit CFG, the named 400s, the OpenAI multipart surface -> a SECOND
# short-lived server on a towerless pack refusing edits by name -> unload.
#
# SKIPs without a pack (tests/convert_qwen_image21_weights.py). Few steps:
# only the wire contract is asserted, never picture quality.
#
# Usage: QWEN_IMAGE_MODEL=<dir> ./tests/test_qwen_image_gen.sh [port]
#        QWEN_IMAGE_NOTOWER_MODEL=<dir> — pack without model.visual.* for the
#        400 arm (default: the tiny fixture pack under ~/claude-tmp)
set -uo pipefail
PORT="${1:-11398}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${QWEN_IMAGE_MODEL:-$(ls -d /Users/Shared/mlx-serve/ddalcu/Qwen-Image-2.1-MLX-Serve-* ~/.mlx-serve/models/ddalcu/Qwen-Image-2.1-MLX-Serve-* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] && [ -f "$MODEL/config.json" ] || { echo "SKIP: no Qwen-Image-2.1 pack (set QWEN_IMAGE_MODEL)"; exit 0; }
NOTOWER="${QWEN_IMAGE_NOTOWER_MODEL:-$HOME/claude-tmp/qwen21-edit/run1/pack-notower}"

OUT="$(mktemp -d)"
LOG="$OUT/server.log"
"$BIN" --serve --model-dir "$OUT" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
SRV2=""
trap 'kill $SRV $SRV2 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done
FAILS=0
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILS=$((FAILS + 1)); }
ID="$(basename "$MODEL")"
gen() { # gen <out.json> <json fields> -> http code
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"size\":\"512x512\",\"steps\":6,\"seed\":3,$2}" -o "$1" -w '%{http_code}'
}
genq() { # genq <out.json> <json fields> — NO injected size: the qwen edit path
  # derives output dims from the LAST condition image when none is requested
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"steps\":6,\"seed\":3,$2}" -o "$1" -w '%{http_code}'
}
png_check() { python3 - "$1" "${2:-2}" <<'PY'
import sys, json, base64, struct
raw = open(sys.argv[1]).read()
if raw.startswith("data:"):
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    replies = [event for event in events if event.get("type") == "complete"]
    assert len(replies) == 1, "missing or duplicate completion"
    reply = replies[0]
else:
    reply = json.loads(raw)
b = base64.b64decode(reply["data"][0]["b64_json"])
assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
assert struct.unpack(">II", b[16:24]) == (512, 512), "wrong size"
assert b[25] == int(sys.argv[2]), "wrong PNG color type (2=RGB, 6=RGBA)"
PY
}
png_dims() { python3 - "$1" <<'PY'
import sys, json, base64, struct
b = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
print("%dx%d" % struct.unpack(">II", b[16:24]))
PY
}

curl -s "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\"}" >/dev/null
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
m = [x for x in json.load(sys.stdin)['data'] if x['id'] == '$ID']
assert m and m[0]['state'] == 'ready' and 'image' in m[0]['capabilities'], m" \
  && pass "load by path -> ready with the image capability" || fail "pack did not load as an image model"
grep -q "\[image\] Qwen-Image-2.1 ready" "$LOG" && pass "qwen_image backend engaged" || fail "no backend ready line"

# Loaded and unloaded rows must both report the pack's real model_type, never
# the image modality's "flux2" marker (a client watching the list would see
# the architecture flip on load).
arch_row() { # arch_row <label>
  curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
m = [x for x in json.load(sys.stdin)['data'] if x['id'] == '$ID']
assert m and m[0].get('meta', {}).get('architecture') == 'qwen_image21', m" \
    && pass "models row reports qwen_image21 ($1)" || fail "models row reports the flux2 modality marker ($1)"
}
arch_row loaded

[ "$(gen "$OUT/a.json" '"prompt":"a red fox in the snow"')" = 200 ] && png_check "$OUT/a.json" \
  && pass "txt2img -> 512x512 PNG" || fail "txt2img"
arch_row after-generate
grep -q "one forward per step" "$LOG" && pass "guidance 1.0 runs one forward per step" || fail "no one-forward log line"

if [[ "$MODEL" == *4bit* ]]; then
  code=$(curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"size\":\"768x768\",\"steps\":2,\"seed\":3,\"prompt\":\"a red fox in the snow\"}" \
    -o "$OUT/wide.json" -w '%{http_code}')
  [ "$code" = 200 ] && python3 - "$OUT/wide.json" <<'PY'
import sys, json, base64, struct
b = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", b[16:24]) == (768, 768)
PY
  png_ok=$?
  [ "$code" = 200 ] && [ "$png_ok" = 0 ] && grep -q '\[mf-linear\] dq-gemm engaged (rows=' "$LOG" \
    && pass "Q4 wide GEMM engages at 768x768" || fail "Q4 wide GEMM did not engage"
fi

[ "$(gen "$OUT/b.json" '"prompt":"a red fox in the snow","guidance_scale":4,"negative_prompt":"blurry"')" = 200 ] && png_check "$OUT/b.json" \
  && pass "guided txt2img -> PNG" || fail "guided txt2img"
grep -q "two forwards per step" "$LOG" && pass "CFG engaged" || fail "CFG did not engage"
cmp -s "$OUT/a.json" "$OUT/b.json" && fail "guidance did not change the render" || pass "guidance changes the render"

python3 - "$OUT/a.json" "$OUT/src.b64" <<'PY'
import sys, json
open(sys.argv[2], "w").write(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
PY
[ "$(gen "$OUT/c.json" "\"prompt\":\"a red fox at night\",\"strength\":0.4,\"image\":\"$(cat "$OUT/src.b64")\"")" = 200 ] && png_check "$OUT/c.json" \
  && pass "img2img -> PNG" || fail "img2img"
grep -q "img2img" "$LOG" && pass "img2img engaged" || fail "no img2img log line"

# ── edit sources: python-built PNGs at known aspects (the output-dims chain
# reads each condition image's native size; the LAST one owns the output)
python3 - "$OUT/src43.png" "$OUT/ref256.png" "$OUT/refp.png" <<'PY'
import sys, struct, zlib

def png(path, W, H, vertical):
    rows = b""
    for y in range(H):
        row = bytearray([0])
        for x in range(W):
            t = (y if vertical else x) * 220 // max(W, H)
            row += bytes([min(255, 30 + t // 2), min(255, 50 + t), min(255, 90 + t // 3)])
        rows += bytes(row)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b""))

png(sys.argv[1], 512, 384, False)  # 4:3 landscape — the edited subject
png(sys.argv[2], 256, 256, True)   # square first extra reference
png(sys.argv[3], 384, 512, True)   # 3:4 portrait — the LAST reference
PY
for n in src43 ref256 refp; do
  python3 -c 'import base64,sys; open(sys.argv[2],"w").write(base64.b64encode(open(sys.argv[1],"rb").read()).decode())' "$OUT/$n.png" "$OUT/$n.b64"
done

# ── instruction edit happy path. A VALID strength rides along on purpose:
# edit mode takes no strength — it must be accepted and ignored, never refused.
[ "$(gen "$OUT/e1.json" "\"prompt\":\"add a small red hat to the subject\",\"mode\":\"edit\",\"strength\":0.5,\"image\":\"$(cat "$OUT/src43.b64")\"")" = 200 ] \
  && [ "$(png_dims "$OUT/e1.json")" = "512x512" ] \
  && pass "instruction edit -> 512x512 PNG (valid strength accepted+ignored)" || fail "instruction edit"
grep -q "\[qwen-image\] edit 512x512 refs=1 steps=6 guidance=1.0 refres=1024 (one forward per step)" "$LOG" \
  && pass "edit engaged (one forward per step)" || fail "no edit engagement line"
grep -q "\[image\] edit: reference .* bytes (byte-based backend)" "$LOG" \
  && pass "byte-based edit transport engaged" || fail "no byte-based edit reference line"

# ── no size: the output follows the LAST reference's aspect at 1024², snapped
# /32 — the pinned qwenEditDims chain (4:3 -> 1184x896 EXACT; a floor would
# say 1152x864)
[ "$(genq "$OUT/e2.json" "\"prompt\":\"add a small red hat to the subject\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src43.b64")\"")" = 200 ] \
  && [ "$(png_dims "$OUT/e2.json")" = "1184x896" ] \
  && pass "sizeless edit keeps the 4:3 source's aspect -> 1184x896" || fail "sizeless edit aspect"
grep -q "edit: target 1024x1024 -> 1184x896 (last reference is 512x384, size matched to source)" "$LOG" \
  && pass "edit target resolved from the last reference (matched to source)" || fail "no edit target line"

# ── multi-reference edit: image + 2 refs (3 total, under the cap). The LAST
# reference (3:4 portrait) owns the output aspect — 896x1184, not the primary's.
[ "$(genq "$OUT/e3.json" "\"prompt\":\"compose the subject and the two references into one image\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src43.b64")\",\"ref_images\":[\"$(cat "$OUT/ref256.b64")\",\"$(cat "$OUT/refp.b64")\"]")" = 200 ] \
  && [ "$(png_dims "$OUT/e3.json")" = "896x1184" ] \
  && pass "multi-ref edit -> 896x1184 (LAST reference's aspect)" || fail "multi-ref edit"
grep -q "\[qwen-image\] edit 896x1184 refs=3 steps=6 guidance=1.0 refres=1024 (one forward per step)" "$LOG" \
  && pass "multi-ref edit engaged (refs=3)" || fail "no multi-ref engagement line"
grep -q "\[image\] edit ref 3: .* bytes (byte-based backend)" "$LOG" \
  && pass "second extra reference engaged (edit ref 3)" || fail "no edit ref 3 line"

# ── the cap: image + 10 dummy refs (11 total) -> 400 by name BEFORE any
# denoise (qwen's cap is 10 total — the other editors' is 4)
code="$(gen "$OUT/e4.json" "\"prompt\":\"x\",\"mode\":\"edit\",\"image\":\"QQ==\",\"ref_images\":[\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\"]")"
[ "$code" = 400 ] && grep -q "too many reference images" "$OUT/e4.json" && grep -q "at most 9" "$OUT/e4.json" \
  && pass "image + 10 ref_images -> 400 (cap: 10 total)" || fail "over-cap edit returned $code"

# ── five references pass: image + 4 tiny PNG refs (5 total, under the cap)
code="$(curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"size\":\"256x256\",\"steps\":6,\"seed\":3,\"prompt\":\"compose the subject and all four references into one image\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src43.b64")\",\"ref_images\":[\"$(cat "$OUT/ref256.b64")\",\"$(cat "$OUT/refp.b64")\",\"$(cat "$OUT/ref256.b64")\",\"$(cat "$OUT/refp.b64")\"]}" \
  -o "$OUT/e4b.json" -w '%{http_code}')"
[ "$code" = 200 ] && [ "$(png_dims "$OUT/e4b.json")" = "256x256" ] \
  && pass "5-reference edit -> 256x256 PNG" || fail "5-ref edit returned $code"

# ── ref_resolution: the per-request conditioning knob (diffusers'
#    output_resolution). 512 engages; out of range is a named 400.
code="$(curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"size\":\"256x256\",\"steps\":6,\"seed\":3,\"prompt\":\"add a red hat\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src43.b64")\",\"ref_resolution\":512}" \
  -o "$OUT/e9.json" -w '%{http_code}')"
[ "$code" = 200 ] && [ "$(png_dims "$OUT/e9.json")" = "256x256" ] && grep -q "refres=512" "$LOG" \
  && pass "ref_resolution 512 edit -> PNG, engagement logged" || fail "ref_resolution 512 edit returned $code"
code="$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"size\":\"256x256\",\"steps\":6,\"seed\":3,\"prompt\":\"add a red hat\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src43.b64")\",\"ref_resolution\":1536}" \
  -o "$OUT/e10.json" -w '%{http_code}')"
[ "$code" = 400 ] && grep -q "ref_resolution.*\[256,1024\]" "$OUT/e10.json" \
  && pass "ref_resolution 1536 -> 400 (named range)" || fail "ref_resolution out-of-range returned $code"
grep -q "\[qwen-image\] edit 256x256 refs=5 steps=6 guidance=1.0 refres=1024 (one forward per step)" "$LOG" \
  && pass "5-ref edit engaged (refs=5)" || fail "no 5-ref engagement line"

# ── the request-scope residency bill: a joint sequence past any working set
#    refuses BY NAME (the #496 regime: high ref counts used to hang the first
#    denoise step with no error). 10 refs + a 2048x2048 target bills >100 GB.
code="$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"size\":\"2048x2048\",\"steps\":6,\"seed\":3,\"prompt\":\"compose\",\"mode\":\"edit\",\"image\":\"QQ==\",\"ref_images\":[\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\",\"QQ==\"]}" \
  -o "$OUT/e11.json" -w '%{http_code}')"
[ "$code" = 400 ] && grep -q "working set" "$OUT/e11.json" \
  && pass "over-budget edit -> 400 named (bill vs headroom)" || fail "over-budget edit returned $code"

# ── edit CFG: guidance 2.5 + negative prompt runs TWO forwards per step and
# changes the render (same seed/steps/source/prompt as the happy path — only
# the guidance differs)
[ "$(gen "$OUT/e5.json" "\"prompt\":\"add a small red hat to the subject\",\"mode\":\"edit\",\"guidance_scale\":2.5,\"negative_prompt\":\"blurry, low quality\",\"image\":\"$(cat "$OUT/src43.b64")\"")" = 200 ] \
  && [ "$(png_dims "$OUT/e5.json")" = "512x512" ] \
  && pass "guided edit -> PNG" || fail "guided edit"
grep -q "\[qwen-image\] edit 512x512 refs=1 steps=6 guidance=2.5 refres=1024 (two forwards per step)" "$LOG" \
  && pass "edit CFG engaged (two forwards per step)" || fail "edit CFG did not engage"
cmp -s "$OUT/e1.json" "$OUT/e5.json" && fail "edit guidance did not change the render" || pass "edit guidance changes the render"

# ── edit 400s: out-of-range strength, no image, cond_weights (0 layer taps)
[ "$(gen "$OUT/e6.json" "\"prompt\":\"x\",\"mode\":\"edit\",\"strength\":1.5,\"image\":\"$(cat "$OUT/src43.b64")\"")" = 400 ] \
  && pass "out-of-range strength in edit mode -> 400" || fail "edit strength 1.5 was not refused"
[ "$(gen "$OUT/e7.json" '"prompt":"x","mode":"edit"')" = 400 ] \
  && pass "edit mode without an image -> 400" || fail "imageless edit was not refused"
[ "$(gen "$OUT/e8.json" '"prompt":"x","cond_weights":"1 1 1"')" = 400 ] \
  && pass "cond_weights is a 400" || fail "cond_weights was not refused"

for stream in false true; do
  [ "$(gen "$OUT/rgba-$stream.json" "\"prompt\":\"This is an RGBA image with transparency. A red apple. The image has alpha channel and the background is transparent.\",\"transparent\":true,\"stream\":$stream")" = 200 ] && png_check "$OUT/rgba-$stream.json" 6 \
    && pass "transparent -> RGBA PNG (stream=$stream)" || fail "transparent (stream=$stream)"
done
[ "$(gen "$OUT/rgb.json" '"prompt":"a red fox in the snow","transparent":false')" = 200 ] && png_check "$OUT/rgb.json" \
  && pass "transparent=false restores RGB after RGBA" || fail "explicit RGB after RGBA"
# ── OpenAI multipart surface: the second file becomes a ref_images entry; the
# sampling knobs (steps et al) ride through the form, not silently dropped
code="$(curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/v1/images/edits" \
  -F "model=$ID" -F "prompt=compose the two pictures into one image" \
  -F "image=@$OUT/src43.png;type=image/png" \
  -F "image[]=@$OUT/refp.png;type=image/png" \
  -F "size=256x256" -F "steps=6" -F "ref_resolution=512" -o "$OUT/mp.json" -w '%{http_code}')"
[ "$code" = 200 ] && [ "$(png_dims "$OUT/mp.json")" = "256x256" ] \
  && pass "/v1/images/edits (multipart, 2 files) -> 256x256 PNG" || fail "multipart edit returned $code"
grep -q "\[qwen-image\] edit 256x256 refs=2 steps=6 guidance=1.0 refres=512 (one forward per step)" "$LOG" \
  && pass "multipart carried both files + the sampling knobs (refs=2 steps=6 refres=512)" || fail "multipart edit lost a reference or a knob"

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && pass "server alive" || fail "server died"
grep -q "\[mlx\]" "$LOG" && fail "MLX error in the log"
curl -s "http://127.0.0.1:$PORT/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null

# ── a pack WITHOUT the vision tower refuses edits BY NAME: a second,
# short-lived server on the tiny fixture pack (boots in seconds)
if [ -f "$NOTOWER/config.json" ]; then
  PORT2=$((PORT + 1))
  LOG2="$OUT/server-notower.log"
  "$BIN" --model "$NOTOWER" --serve --port "$PORT2" >"$LOG2" 2>&1 &
  SRV2=$!
  up2=""
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT2/health" >/dev/null 2>&1 && { up2=1; break; }
    kill -0 $SRV2 2>/dev/null || break
    sleep 1
  done
  if [ -n "$up2" ]; then
    code="$(curl -s "http://127.0.0.1:$PORT2/v1/images/generations" -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"add a small red hat\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src43.b64")\"}" \
      -o "$OUT/nt.json" -w '%{http_code}')"
    if [ "$code" = 400 ] && grep -q "vision tower" "$OUT/nt.json"; then
      pass "towerless pack refuses edits by name (vision tower)"
    else
      fail "towerless edit returned $code: $(head -c 200 "$OUT/nt.json")"
    fi
  else
    fail "towerless server did not start"
  fi
  kill $SRV2 2>/dev/null
  SRV2=""
else
  echo "SKIP: no towerless pack for the 400 arm (set QWEN_IMAGE_NOTOWER_MODEL)"
fi

[ "$FAILS" = 0 ] && echo "ALL PASS" || { echo "$FAILS FAILED (log: $LOG)"; exit 1; }
