#!/usr/bin/env bash
# test_moss_image_gen.sh — MOSS-Image on the diffusers pack: load by path (`image` capability + `MOSS-Image ready`), a
# 512x512 PNG at the default CFG 3 by ENGAGEMENT (`guidance=3.0 (two forwards per step`), guidance_scale 1.0 running
# one forward and rendering differently, a layout JSON prompt with the recommended negative tags accepted, img2img and
# edit as named 400s, unload, alive + no MLX error. 8 steps: the wire contract, never picture quality.
#
#   MOSS_IMAGE_MODEL=<dir> ./tests/test_moss_image_gen.sh [port]   # default mlx-community/MOSS-Image-SFT-*
#
# SKIPs without a pack. Parity: tests/dump_moss_image_fixtures.py + MOSS_IMAGE_TEST_MODEL / MOSS_IMAGE_FIXTURE.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11399}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${MOSS_IMAGE_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/MOSS-Image-* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] && [ -f "$MODEL/model_index.json" ] || { echo "SKIP: no MOSS-Image pack (set MOSS_IMAGE_MODEL)"; exit 0; }

OUT="$(mktemp -d)"
LOG="$OUT/server.log"
"$BIN" --serve --model-dir "$OUT" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
srv_wait_health "http://127.0.0.1:$PORT" 60 1 "$SRV" || kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
FAILS=0
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILS=$((FAILS + 1)); }
ID="$(basename "$MODEL")"
gen() { # gen <out.json> <json fields> -> http code
  curl -s -m 3600 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID\",\"size\":\"512x512\",\"steps\":8,\"seed\":3,$2}" -o "$1" -w '%{http_code}'
}
png_check() { python3 - "$1" <<'PY'
import sys, json, base64, struct
b = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
assert struct.unpack(">II", b[16:24]) == (512, 512), "wrong size"
PY
}

curl -s "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\"}" >/dev/null
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
m = [x for x in json.load(sys.stdin)['data'] if x['id'] == '$ID']
assert m and m[0]['state'] == 'ready' and 'image' in m[0]['capabilities'], m" \
  && pass "load by path -> ready with the image capability" || fail "pack did not load as an image model"
grep -q "MOSS-Image ready" "$LOG" && pass "backend ready line" || fail "no MOSS-Image ready line"

code="$(gen "$OUT/cfg3.json" '"prompt":"A red fox in fresh snow, golden hour"')"
[ "$code" = 200 ] && png_check "$OUT/cfg3.json" && pass "txt2img 512x512 PNG" || fail "txt2img (HTTP $code)"
grep -q "guidance=3.0 (two forwards per step" "$LOG" && pass "default CFG 3 engages" || fail "default CFG did not engage"

code="$(gen "$OUT/cfg1.json" '"prompt":"A red fox in fresh snow, golden hour","guidance_scale":1.0')"
[ "$code" = 200 ] && grep -q "guidance=1.0 (one forward per step)" "$LOG" && pass "guidance 1.0 runs one forward" || fail "guidance 1.0 (HTTP $code)"
cmp -s "$OUT/cfg3.json" "$OUT/cfg1.json" && fail "CFG did not change the render" || pass "CFG changes the render"

LAYOUT='{\"summary\":{\"meta\":[\"poster\"],\"canvas\":\"single\",\"caption\":\"A minimalist travel poster.\"},\"items\":[{\"label\":\"text\",\"caption\":\"Large title.\",\"text\":\"EXPLORE\",\"bbox\":[-208,-192,-144,192]}],\"bbox_coord_space\":\"abs_centralized\",\"bbox_format\":\"yxyx\"}'
code="$(gen "$OUT/layout.json" "\"prompt\":\"$LAYOUT\",\"negative_prompt\":\"nsfw, worst quality, bad anatomy, watermark, jpeg artifact\"")"
[ "$code" = 200 ] && png_check "$OUT/layout.json" && pass "layout prompt + negative" || fail "layout prompt (HTTP $code)"

IMG="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['data'][0]['b64_json'])" "$OUT/cfg3.json")"
code="$(gen "$OUT/i2i.json" "\"prompt\":\"x\",\"image\":\"$IMG\"")"
[ "$code" = 400 ] && pass "img2img refused by name" || fail "img2img not refused (HTTP $code)"
code="$(gen "$OUT/edit.json" "\"prompt\":\"x\",\"mode\":\"edit\",\"image\":\"$IMG\"")"
[ "$code" = 400 ] && pass "edit refused by name" || fail "edit not refused (HTTP $code)"

curl -s "http://127.0.0.1:$PORT/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
kill -0 $SRV 2>/dev/null && pass "server alive" || fail "server died"
grep -q "MLX error" "$LOG" && fail "MLX error in the log" || pass "no MLX error"
grep "\[moss-image\] stages" "$LOG" | tail -1
[ "$FAILS" = 0 ] && echo "ALL PASS" || { echo "$FAILS FAILED (log: $LOG)"; exit 1; }
