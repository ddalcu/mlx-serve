#!/usr/bin/env bash
# Qwen-Image-2.1 on the ONE main server: headless boot -> load the converted
# pack by absolute path -> txt2img PNG of the requested size -> real CFG engages
# (two forwards per step, by the log) and changes the render -> img2img -> the
# named 400s for what the backend cannot honor -> unload.
#
# SKIPs without a pack (tests/convert_qwen_image21_weights.py). Few steps at
# 512x512: only the wire contract is asserted, never picture quality.
#
# Usage: QWEN_IMAGE_MODEL=<dir> ./tests/test_qwen_image_gen.sh [port]
set -uo pipefail
PORT="${1:-11398}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${QWEN_IMAGE_MODEL:-$(ls -d /Users/Shared/mlx-serve/ddalcu/Qwen-Image-2.1-MLX-Serve-* ~/.mlx-serve/models/ddalcu/Qwen-Image-2.1-MLX-Serve-* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] && [ -f "$MODEL/config.json" ] || { echo "SKIP: no Qwen-Image-2.1 pack (set QWEN_IMAGE_MODEL)"; exit 0; }

OUT="$(mktemp -d)"
LOG="$OUT/server.log"
"$BIN" --serve --model-dir "$OUT" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
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

curl -s "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\"}" >/dev/null
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
m = [x for x in json.load(sys.stdin)['data'] if x['id'] == '$ID']
assert m and m[0]['state'] == 'ready' and 'image' in m[0]['capabilities'], m" \
  && pass "load by path -> ready with the image capability" || fail "pack did not load as an image model"
grep -q "\[image\] Qwen-Image-2.1 ready" "$LOG" && pass "qwen_image backend engaged" || fail "no backend ready line"

[ "$(gen "$OUT/a.json" '"prompt":"a red fox in the snow"')" = 200 ] && png_check "$OUT/a.json" \
  && pass "txt2img -> 512x512 PNG" || fail "txt2img"
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

[ "$(gen "$OUT/d.json" "\"prompt\":\"x\",\"mode\":\"edit\",\"image\":\"$(cat "$OUT/src.b64")\"")" = 400 ] \
  && pass "edit mode is a 400" || fail "edit mode was not refused"
[ "$(gen "$OUT/e.json" '"prompt":"x","cond_weights":"1 1 1"')" = 400 ] \
  && pass "cond_weights is a 400" || fail "cond_weights was not refused"

for stream in false true; do
  [ "$(gen "$OUT/rgba-$stream.json" "\"prompt\":\"This is an RGBA image with transparency. A red apple. The image has alpha channel and the background is transparent.\",\"transparent\":true,\"stream\":$stream")" = 200 ] && png_check "$OUT/rgba-$stream.json" 6 \
    && pass "transparent -> RGBA PNG (stream=$stream)" || fail "transparent (stream=$stream)"
done
[ "$(gen "$OUT/rgb.json" '"prompt":"a red fox in the snow","transparent":false')" = 200 ] && png_check "$OUT/rgb.json" \
  && pass "transparent=false restores RGB after RGBA" || fail "explicit RGB after RGBA"

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && pass "server alive" || fail "server died"
grep -q "\[mlx\]" "$LOG" && fail "MLX error in the log"
curl -s "http://127.0.0.1:$PORT/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
[ "$FAILS" = 0 ] && echo "ALL PASS" || { echo "$FAILS FAILED (log: $LOG)"; exit 1; }
