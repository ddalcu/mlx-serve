#!/usr/bin/env bash
# NSFW content-filter on /v1/images/generations: with the threshold forced to 0
# (every image flagged) the filter must BLOCK by default, a per-request
# "safety":false must BYPASS, and --no-safety must disable it process-wide.
# Validates the block path + both opt-outs without needing actual NSFW content
# (the classifier itself is proven bit-exact by the src/nsfw.zig oracle).
#
# Needs the Falconsai classifier at ~/.mlx-serve/models/Falconsai/nsfw_image_detection
# and an image model. Usage: KREA_MODEL=<dir> ./tests/test_nsfw_filter.sh [port]
set -uo pipefail
PORT="${1:-11400}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

IMG="${KREA_MODEL:-${FLUX_MODEL:-$(ls -d ~/.mlx-serve/models/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8 2>/dev/null | head -1)}}"
[ -n "$IMG" ] || { echo "SKIP: no image model (set KREA_MODEL or FLUX_MODEL)"; exit 0; }
[ -f ~/.mlx-serve/models/Falconsai/nsfw_image_detection/model.safetensors ] || { echo "SKIP: NSFW classifier not downloaded"; exit 0; }
IMG_ID="$(basename "$IMG")"
HUB=~/.cache/huggingface/hub

start() {  # $1 = extra flags
    pkill -f "mlx-serve.*$PORT" 2>/dev/null; sleep 1
    MLX_SERVE_NSFW_THRESHOLD=0 "$BIN" --serve --model-dir "$HUB" --port "$PORT" $1 >/tmp/test_nsfw_server.log 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 /tmp/test_nsfw_server.log; exit 1; }
        sleep 1
    done
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/load-model" -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$IMG\"}" >/dev/null
}
gen() {  # $1 = extra json fields → prints HTTP status
    curl -s -m 1200 -o /tmp/test_nsfw_out.json -w '%{http_code}' "http://127.0.0.1:$PORT/v1/images/generations" \
        -X POST -H 'Content-Type: application/json' \
        -d "{\"model\":\"$IMG_ID\",\"prompt\":\"a red fox in the snow\",\"size\":\"512x512\",\"steps\":8$1}"
}

# 1. Filter ON (default), threshold 0 → every image blocked → 400.
start ""
trap 'kill $SRV 2>/dev/null' EXIT
CODE=$(gen "")
[ "$CODE" = "400" ] || { echo "FAIL: expected 400 (blocked), got $CODE: $(cat /tmp/test_nsfw_out.json)"; exit 1; }
grep -q "content filter" /tmp/test_nsfw_out.json || { echo "FAIL: 400 but no content-filter message"; exit 1; }
echo "PASS: filter blocks flagged output (400)"

# 2. Per-request "safety":false → bypass → 200 PNG.
CODE=$(gen ",\"safety\":false")
[ "$CODE" = "200" ] || { echo "FAIL: safety:false should bypass (expected 200), got $CODE"; exit 1; }
python3 - /tmp/test_nsfw_out.json <<'PY'
import sys,json,base64
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
print("PASS: \"safety\":false bypasses the filter (200 PNG)")
PY

# 3. --no-safety process-wide → bypass → 200 PNG.
kill $SRV 2>/dev/null
start "--no-safety"
CODE=$(gen "")
[ "$CODE" = "200" ] || { echo "FAIL: --no-safety should disable the filter (expected 200), got $CODE"; exit 1; }
echo "PASS: --no-safety disables the filter (200)"

echo "ALL PASS: NSFW content filter (block default, per-request + --no-safety opt-outs)"
