#!/bin/bash
# test_vision_chunked_prefill.sh — an image prompt prefills in chunks like text (the guard bills the chunk width; an
# unchunked image turn 400'd long conversations). Two arms of one binary, --prefill-chunk 8192 (one chunk) vs 32, with
# the prompt padded past TAIL_MERGE_MAX so boundaries land INSIDE the image span: [1] only the 32 arm logs
# `[vision] chunked prefill`; [2] both arms name the same quadrant colors (a splice that restarts its row index per
# chunk loses the bottom quadrants). Byte equality across arms is NOT asserted: chunk width changes GEMM shapes and
# 4-bit near-ties flip.
#
#   VISION_CHUNK_TEST_MODEL=<vision model> ./tests/test_vision_chunked_prefill.sh [port]   # default LFM2.5-VL-1.6B
#
# SKIPs without a vision model. Hermetic twins: spliceVisionRows chunk / audio / video tests, countSpliceRows.

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11499}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${VISION_CHUNK_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/LFM2.5-VL-1.6B-4bit}"
WORK="$(mktemp -d)"
SERVER_PID=""

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "[skip] no vision model at $MODEL (set VISION_CHUNK_TEST_MODEL)"
    exit 0
fi

# A deterministic quadrant PNG (red TL, blue TR, green BL, yellow BR) — the
# answer depends on reading the WHOLE image, so losing the later vision rows
# is visible in the reply. Filler pads the prompt past TAIL_MERGE_MAX.
python3 - "$WORK" <<'PYEOF'
import base64, json, struct, sys, zlib

work = sys.argv[1]
W = H = 224
rows = b""
for y in range(H):
    row = b"\x00"
    for x in range(W):
        if x < W // 2 and y < H // 2:   px = (220, 30, 30)    # red TL
        elif x >= W // 2 and y < H // 2: px = (30, 30, 220)   # blue TR
        elif x < W // 2:                 px = (30, 200, 30)   # green BL
        else:                            px = (230, 220, 40)  # yellow BR
        row += bytes(px)
    rows += row
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows))
       + chunk(b"IEND", b""))
b64 = base64.b64encode(png).decode()

filler = " ".join(f"item{i} shelf {i % 7}." for i in range(120))
body = {
    "model": "mlx-serve",
    "temperature": 0.0,
    "max_tokens": 96,
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": "Notes: " + filler + "\nIgnoring the notes: name the color of each quadrant of the image, top-left first."},
        ]},
    ],
}
with open(f"{work}/req.json", "w") as f:
    json.dump(body, f)
PYEOF

run_arm() {
    local arm="$1" chunk="$2"
    local log="$WORK/$arm.log"
    "$BINARY" --serve --model "$MODEL" --prefill-chunk "$chunk" \
        --host 127.0.0.1 --port "$PORT" --prefix-cache-entries 0 \
        --log-level debug --log-file "$log" \
        >"$WORK/$arm.stdout" 2>&1 &
    SERVER_PID=$!
    local ready=0
    srv_wait_health "http://127.0.0.1:$PORT" 240 0.5 && ready=1
    if [ "$ready" -ne 1 ]; then
        echo "[fail] server did not start ($arm arm)"
        exit 1
    fi
    curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary @"$WORK/req.json" > "$WORK/$arm.resp.json" \
        || { echo "[fail] request failed ($arm arm)"; exit 1; }
    sleep 1
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    sleep 1
}

run_arm off 8192
run_arm on 32

OFF_TXT=$(python3 -c "import json; print(json.load(open('$WORK/off.resp.json'))['choices'][0]['message']['content'])")
ON_TXT=$(python3 -c "import json; print(json.load(open('$WORK/on.resp.json'))['choices'][0]['message']['content'])")

echo "off: $OFF_TXT"
echo "on:  $ON_TXT"

if ! grep -q '\[vision\] chunked prefill' "$WORK/on.log"; then
    echo "[fail] on arm never logged [vision] chunked prefill — vision still runs one whole-prompt forward"
    exit 1
fi
if grep -q '\[vision\] chunked prefill' "$WORK/off.log"; then
    echo "[fail] the one-chunk arm logged [vision] chunked prefill — its prompt did not fit one chunk"
    exit 1
fi
# Perceived-color set per arm. Checkpoint-agnostic bar: a model may call the
# dark-yellow quadrant "brown"/"orange" (gemma-4-qat does), but BOTH arms see
# the same image, so their color sets must match — the class bug (splice rows
# restarting at a chunk boundary) collapses the bottom quadrants onto the top
# colors, shrinking the on arm's set. The off arm must still read the top row
# (red + blue) and >= 3 distinct colors, or the model isn't reading the image
# at all and the set comparison proves nothing.
color_set() {
    echo "$1" | tr '[:upper:]' '[:lower:]' \
        | grep -oE 'red|blue|green|yellow|brown|orange|purple|gray|grey|white|black|cyan|magenta|gold|teal|pink' \
        | sort -u | tr '\n' ' '
}
OFF_SET=$(color_set "$OFF_TXT")
ON_SET=$(color_set "$ON_TXT")
echo "off colors: $OFF_SET"
echo "on colors:  $ON_SET"
case "$OFF_SET" in
    *red*) : ;;
    *) echo "[fail] off arm never names red — model is not reading the image"; exit 1 ;;
esac
case "$OFF_SET" in
    *blue*) : ;;
    *) echo "[fail] off arm never names blue — model is not reading the image"; exit 1 ;;
esac
if [ "$(echo "$OFF_SET" | wc -w)" -lt 3 ]; then
    echo "[fail] off arm names fewer than 3 colors — model is not reading the image"
    exit 1
fi
if [ "$ON_SET" != "$OFF_SET" ]; then
    echo "[fail] chunked arm perceives different colors than the unchunked arm — vision rows lost"
    exit 1
fi
echo "[pass] vision prefill chunks and the splice is chunk-exact across boundaries"
