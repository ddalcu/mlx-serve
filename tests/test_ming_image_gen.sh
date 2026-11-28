#!/usr/bin/env bash
# test_ming_image_gen.sh — Ming-Image on the release tree (or mlx-community's quantized one): load by path (`image`
# capability + `Ming-Image ready`), a 512x512 RGBA PNG at the default (`one forward per step`), guidance_scale 2
# running two forwards and rendering differently, a non-bucket size snapped to the release's bucket, negative_prompt
# and img2img as named 400s, an edit of the 1280x720 render returned at its aspect (and `size` / `ref_images` on an
# edit as named 400s), unload, alive + no MLX error. 4 steps: the wire contract, never picture quality.
#
#   MING_IMAGE_MODEL=<dir> ./tests/test_ming_image_gen.sh [port]   # default inclusionAI/Ming-Image-*
#
# SKIPs without a pack. Parity: tests/dump_ming_image_fixtures.py + MING_IMAGE_TEST_MODEL / MING_IMAGE_FIXTURE.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11398}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${MING_IMAGE_MODEL:-$(ls -d ~/.mlx-serve/models/*/Ming-Image-*Design* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] && [ -f "$MODEL/transformer/config.json" ] || { echo "SKIP: no Ming-Image pack (set MING_IMAGE_MODEL)"; exit 0; }

OUT="$(mktemp -d "$HOME/claude-tmp/ming-image-gen.XXXX" 2>/dev/null || mktemp -d)"
LOG="$OUT/server.log"
"$BIN" --serve --model-dir "$OUT" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'srv_stop $SRV' EXIT
srv_wait_health "http://127.0.0.1:$PORT" 60 1 "$SRV" || kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
FAILS=0
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILS=$((FAILS + 1)); }
ID="$(basename "$MODEL")"
gen() { # gen <out.json> <json fields> -> http code (the body goes through a file: an image outgrows argv)
  printf '{"model":"%s","steps":4,"seed":3,%s}' "$ID" "$2" >"$1.req"
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d @"$1.req" -o "$1" -w '%{http_code}'
}
png_check() { python3 - "$1" "$2" "$3" <<'PY'
import sys, json, base64, struct
b = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
assert struct.unpack(">II", b[16:24]) == (int(sys.argv[2]), int(sys.argv[3])), struct.unpack(">II", b[16:24])
assert b[25] == 6, "no alpha channel"
PY
}

curl -s "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\"}" >/dev/null
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
m = [x for x in json.load(sys.stdin)['data'] if x['id'] == '$ID']
assert m and m[0]['state'] == 'ready' and 'image' in m[0]['capabilities'], m" \
  && pass "load by path -> ready with the image capability" || fail "pack did not load as an image model"
grep -q "Ming-Image ready" "$LOG" && pass "backend ready line" || fail "no Ming-Image ready line"

P='"prompt":"A flat vector icon of a red fox on a transparent background"'
code="$(gen "$OUT/g1.json" "$P,\"size\":\"512x512\"")"
[ "$code" = 200 ] && png_check "$OUT/g1.json" 512 512 && pass "txt2img 512x512 RGBA PNG" || fail "txt2img (HTTP $code)"
grep -q "guidance=1.00 (one forward per step)" "$LOG" && pass "default guidance: one forward per step" || fail "default guidance did not run one forward"

code="$(gen "$OUT/g2.json" "$P,\"size\":\"512x512\",\"guidance_scale\":2.0")"
[ "$code" = 200 ] && grep -q "guidance=2.00 (two forwards per step)" "$LOG" && pass "guidance 2 runs two forwards" || fail "guidance 2 (HTTP $code)"
cmp -s "$OUT/g1.json" "$OUT/g2.json" && fail "guidance did not change the render" || pass "guidance changes the render"

code="$(gen "$OUT/wide.json" "$P,\"size\":\"1280x720\"")"
[ "$code" = 200 ] && png_check "$OUT/wide.json" 1280 720 && pass "1280x720 is its own bucket" || fail "1280x720 (HTTP $code)"
code="$(gen "$OUT/snap.json" "$P,\"size\":\"1000x1000\"")"
[ "$code" = 200 ] && png_check "$OUT/snap.json" 1024 1024 && pass "1000x1000 snaps to the 1024 bucket" || fail "bucket snap (HTTP $code)"

code="$(gen "$OUT/neg.json" "$P,\"size\":\"512x512\",\"negative_prompt\":\"blurry\"")"
[ "$code" = 400 ] && pass "negative_prompt refused by name" || fail "negative_prompt not refused (HTTP $code)"
IMG="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['data'][0]['b64_json'])" "$OUT/g1.json")"
code="$(gen "$OUT/i2i.json" "\"prompt\":\"x\",\"image\":\"$IMG\"")"
[ "$code" = 400 ] && pass "img2img refused by name" || fail "img2img not refused (HTTP $code)"
WIDE="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['data'][0]['b64_json'])" "$OUT/wide.json")"
E='"prompt":"Make the fox blue","mode":"edit"'
code="$(gen "$OUT/edit.json" "$E,\"image\":\"$WIDE\"")"
[ "$code" = 200 ] && png_check "$OUT/edit.json" 1280 720 && pass "edit returns an RGBA PNG at the picture's aspect" || fail "edit (HTTP $code)"
grep -q "edit 1280x720 -> canvas 1280x720 -> 1280x720" "$LOG" && grep -q "\[ming-image\] edit 1280x720 steps=" "$LOG" \
  && pass "edit ran with the reference frame" || fail "no edit engagement lines"
code="$(gen "$OUT/edit_size.json" "$E,\"image\":\"$WIDE\",\"size\":\"512x512\"")"
[ "$code" = 400 ] && grep -q "keeps the picture's aspect" "$OUT/edit_size.json" && pass "size on an edit refused by name" || fail "size on an edit (HTTP $code)"
code="$(gen "$OUT/edit_refs.json" "$E,\"image\":\"$WIDE\",\"ref_images\":[\"$IMG\"]")"
[ "$code" = 400 ] && grep -q "edits one picture" "$OUT/edit_refs.json" && pass "ref_images refused by name" || fail "ref_images (HTTP $code)"

curl -s "http://127.0.0.1:$PORT/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
kill -0 $SRV 2>/dev/null && pass "server alive" || fail "server died"
grep -q "MLX error" "$LOG" && fail "MLX error in the log" || pass "no MLX error"
grep "\[ming-image\] stages" "$LOG" | tail -3
[ "$FAILS" = 0 ] && echo "ALL PASS" || { echo "$FAILS FAILED (log: $LOG)"; exit 1; }
