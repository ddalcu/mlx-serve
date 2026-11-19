#!/usr/bin/env bash
# test_minimax_h3_image.sh — MiniMax-H3-Image end to end on the converted pack: `image` capability, the staged
# preflight's NUMBER (the text encoder stage binds the 4-bit pack, never the ~37 GB sum), named 400s (a step count, a
# side past 6272, a bad `min_generation_area`), a seeded text-to-image PNG at the requested size that is a picture (no
# seams on the 16-px latent grid), a small request generated on the minimum-area canvas and resized back (logged), and
# an edit of that image that keeps its size, counts one reference and is a picture too.
#
#   H3_IMAGE_MODEL=<dir> ./tests/test_minimax_h3_image.sh [port]   # default ddalcu/MiniMax-H3-Image-MLX-Serve-4bit; SKIP without
#
# Hermetic twins: the `minimax_h3_image` plugin and config tests (tiny pack, canvas and reference sizes, the bill).
source "$(dirname "$0")/_lib_models.sh"

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11363}"
MODEL="${H3_IMAGE_MODEL:-$(find_model ddalcu/MiniMax-H3-Image-MLX-Serve-4bit || echo "$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-Image-MLX-Serve-4bit")}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3-Image pack at $MODEL (set H3_IMAGE_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
OUT="$HOME/claude-tmp/test_minimax_h3_image"
mkdir -p "$OUT"
LOG="$OUT/server.log"

"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'srv_stop "$SRV"' EXIT
srv_wait_health "http://127.0.0.1:$PORT" 180 1 "$SRV" || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
rc=0

# [1] capability + staged preflight: max(TE 15.8, DiT 18.7 shedding AdaLN, VAE 2.3), not their sum.
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"image"' \
  && echo "PASS: /v1/models advertises image" || { echo "FAIL: /v1/models missing image capability"; rc=1; }
PEAK=$(grep -m1 "media peak" "$LOG" | sed -E 's/.*media peak ~([0-9.]+) GB.*/\1/')
if [ -n "$PEAK" ] && python3 -c "import sys; sys.exit(0 if 12.0 < float('$PEAK') < 20.0 else 1)"; then
  echo "PASS: media preflight billed the biggest stage (~$PEAK GB)"
else
  echo "FAIL: media preflight peak '$PEAK' GB is not the staged bill"; rc=1
fi

# post <body> <png-out> -> "WxH" or "ERR <code> <body>"
post() {
  local tmp code
  tmp=$(mktemp "$OUT/resp.XXXX")
  code=$(curl -s --max-time 1800 -o "$tmp" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
         -H 'Content-Type: application/json' -d @- <<<"$1")
  if [ "$code" != "200" ]; then echo "ERR $code $(head -c 200 "$tmp")"; rm -f "$tmp"; return; fi
  python3 - "$tmp" "$2" <<'PY'
import base64, json, struct, sys
png = base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert png[:8] == b"\x89PNG\r\n\x1a\n"
open(sys.argv[2], "wb").write(png)
print("%dx%d" % struct.unpack(">II", png[16:24]))
PY
  rm -f "$tmp"
}

# A decoded latent that never denoised has its edges on the VAE's 16-px grid: the mean gradient across those seams
# over the mean elsewhere is ~2.5 for noise, ~1.0 for a picture. Sets SEAM.
seams_ok() {
  SEAM=$(python3 - "$1" <<'PY'
import struct, sys, zlib
d = open(sys.argv[1], "rb").read()
i, idat, w, h = 8, b"", 0, 0
while i < len(d):
    n = struct.unpack(">I", d[i:i + 4])[0]; t = d[i + 4:i + 8]
    if t == b"IHDR": w, h = struct.unpack(">II", d[i + 8:i + 16])
    if t == b"IDAT": idat += d[i + 8:i + 8 + n]
    i += 12 + n
raw, stride, prev = zlib.decompress(idat), w * 3 + 1, bytearray(w * 3)
seam = rest = ns = nr = 0
for y in range(h):
    f, cur = raw[y * stride], bytearray(raw[y * stride + 1:(y + 1) * stride])
    for x in range(w * 3):
        a = cur[x - 3] if x >= 3 else 0; b = prev[x]; c = prev[x - 3] if x >= 3 else 0
        if f == 1: cur[x] = (cur[x] + a) & 255
        elif f == 2: cur[x] = (cur[x] + b) & 255
        elif f == 3: cur[x] = (cur[x] + ((a + b) >> 1)) & 255
        elif f == 4:
            p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            cur[x] = (cur[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
    prev = cur
    if y % 4: continue
    for x in range(1, w):
        g = abs(cur[3 * x + 1] - cur[3 * x - 2])
        if x % 16 == 0: seam += g; ns += 1
        else: rest += g; nr += 1
print("%.2f" % ((seam / ns) / max(rest / nr, 1e-6)))
PY
)
  python3 -c "import sys; sys.exit(0 if float('$SEAM') < 1.3 else 1)"
}

# [2] named 400s, before any stage loads.
for c in '{"prompt":"x","steps":4}|omit '"'"'steps'"'"'' '{"prompt":"x","size":"8000x512"}|6272' '{"prompt":"x","min_generation_area":5}|min_generation_area'; do
  body=${c%%|*}; needle=${c#*|}
  r=$(post "$body" "$OUT/none.png")
  case "$r" in ("ERR 400"*"$needle"*) echo "PASS: $body -> named 400";; (*) echo "FAIL: $body -> $r"; rc=1;; esac
done

# [3] text to image at the requested size (the canvas IS the request at min_generation_area 1 MP).
d=$(post '{"prompt":"a red bicycle leaning against a blue door, morning light","size":"1024x768","seed":7,"min_generation_area":786432}' "$OUT/t2i.png")
if [ "$d" = "1024x768" ] && seams_ok "$OUT/t2i.png"; then echo "PASS: text to image -> $d, a picture (seam ratio $SEAM)"
else echo "FAIL: text to image -> $d (seam ratio ${SEAM:-n/a}: decoded noise has edges on the 16-px latent grid)"; rc=1; fi
grep -q "\[minimax-h3-image\] step 6/6" "$LOG" && echo "PASS: six PDD steps ran" || { echo "FAIL: no step 6/6 line"; rc=1; }

# [4] a request under the minimum area is generated larger at its aspect and resized back.
d=$(post '{"prompt":"a lighthouse on a cliff","size":"512x384","seed":3,"min_generation_area":786432}' "$OUT/small.png")
[ "$d" = "512x384" ] && grep -q "512x384 generated on a 1024x768 canvas" "$LOG" \
  && echo "PASS: 512x384 generated on the 1024x768 canvas, returned at 512x384" || { echo "FAIL: small request -> $d"; rc=1; }

# [5] an edit of [3]: keeps the source size, one reference through the encoder and the tower.
if [ -f "$OUT/t2i.png" ]; then
  SRC_B64=$(python3 -c 'import base64,sys;print(base64.b64encode(open(sys.argv[1],"rb").read()).decode())' "$OUT/t2i.png")
  d=$(post "{\"prompt\":\"Make the bicycle in <Picture 1> bright yellow\",\"mode\":\"edit\",\"image\":\"$SRC_B64\",\"seed\":7,\"min_generation_area\":786432}" "$OUT/edit.png")
  [ "$d" = "1024x768" ] && grep -q "1 reference(s) -> " "$LOG" && seams_ok "$OUT/edit.png" \
    && echo "PASS: edit keeps 1024x768, encodes one reference, a picture (seam ratio $SEAM)" || { echo "FAIL: edit -> $d (seam ratio ${SEAM:-n/a})"; rc=1; }
fi

echo "outputs: $OUT"
exit $rc
