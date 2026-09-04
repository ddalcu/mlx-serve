#!/bin/bash
# /v1/embeddings multimodal (image) end-to-end test — Qwen3-VL-Embedding class.
#
# PR #351 routes any /v1/embeddings input that carries an image (OpenAI
# content-parts array: [{type:"text"...},{type:"image_url"...}]) to the
# vision path: vision-tower encode + DeepStack streams, pad-spliced tokens,
# interleaved M-RoPE, DeepStack injection. This script pins that path from
# the HTTP surface:
#
#   1. image-only request (previously: image silently dropped, no image in
#      the input at all → 400): must return an OpenAI-shaped unit-norm vector
#   2. text+image parts request: same contract
#   3. self-consistency: identical requests → cos = 1.000
#   4. image+text vs the same text alone differ (the image actually
#      contributes to the pooled vector, not silently dropped)
#   5. pure-text request still works on the same server (no regression)
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - MM_TEST_MODEL or ~/.mlx-serve/models/mlx-community/Qwen3-VL-Embedding-2B-4bit
#
# Usage: ./tests/test_embeddings_mm.sh [port]

set -e

PORT=${1:-11330}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MM_MODEL="${MM_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-VL-Embedding-2B-4bit}"
if [ ! -d "$MM_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_embeddings_mm: Qwen3-VL-Embedding model not found at $MM_MODEL"
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

FAILURES=0
check() {
    local desc="$1" ok="$2" detail="$3"
    if [ "$ok" = "1" ]; then
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        echo -e "  ${RED}FAIL${NC} $desc"
        [ -n "$detail" ] && echo "    $detail"
        FAILURES=$((FAILURES + 1))
    fi
}

SERVER_PID=""
start_server() {
    local logfile="$1"; shift
    "$BINARY" --serve --port "$PORT" "$@" > "$logfile" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 120); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -5 "$logfile"; return 1
}
stop_server() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; }
trap 'stop_server' EXIT

echo "=== /v1/embeddings multimodal: $MM_MODEL ==="
start_server /tmp/test_embeddings_mm_server.log --model "$MM_MODEL" --log-level info

# --- 1 + 2 + 3. vision-path contract, one python block over the HTTP API ---
python3 - "$BASE" > /tmp/test_embeddings_mm.out <<'EOF'
import base64, json, math, struct, sys, urllib.request, zlib

base = sys.argv[1]

# 56x56 red PNG, built with the stdlib alone (no PIL). 56 = 28*2 matches the
# Qwen3-VL patch grid (patch 14 x merge 2) so the smart-resize path keeps the
# grid non-empty. Content is irrelevant — the contract is that the image
# participates in the pooled vector at all.
def png_side(width, height, rgb):
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    row = b"\x00" + bytes(rgb) * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(row * height)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))

b64 = base64.b64encode(png_side(56, 56, (255, 0, 0))).decode()
data_uri = "data:image/png;base64," + b64

def post(payload):
    req = urllib.request.Request(base + "/v1/embeddings",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    rows = sorted(r["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in rows]

def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

def shape_ok(vec):
    return len(vec) > 0 and abs(math.sqrt(sum(x*x for x in vec)) - 1.0) < 1e-3

image_part = {"type": "image_url", "image_url": {"url": data_uri}}
text_part = {"type": "text", "text": "A red square."}

results = {}
checks = []

# 1. image-only input (was: silent drop / 400 before the PR)
img_only = post({"model": "mlx-serve", "input": [image_part]})
checks.append(("image-only returns a vector", len(img_only) == 1 and shape_ok(img_only[0])))
results["img_only"] = img_only[0]

# 2. text+image parts (OpenAI shape: ONE flat content-parts array — the
#    nested [[parts]] multi-item form is intentionally rejected with a 400)
txt_img = post({"model": "mlx-serve", "input": [text_part, image_part]})
checks.append(("text+image parts return a vector", len(txt_img) == 1 and shape_ok(txt_img[0])))
results["txt_img"] = txt_img[0]

# 3. self-consistency: same request twice → identical vector
img_only_2 = post({"model": "mlx-serve", "input": [image_part]})
checks.append(("repeated image-only request is self-consistent (cos=1)", cos(img_only[0], img_only_2[0]) > 0.9999))

# 4. the image actually contributes: text+image vs the same text alone
txt_alone = post({"model": "mlx-serve", "input": "A red square."})
d = cos(txt_alone[0], txt_img[0])
checks.append(("text+image differs from text alone (cos=%.4f < 0.999)" % d, d < 0.999))

# 5. pure-text still works on the same server (no regression on the text path)
checks.append(("pure-text request still unit-norm", shape_ok(txt_alone[0])))

for desc, ok in checks:
    print(("PASS\t" if ok else "FAIL\t") + desc)
sys.exit(0 if all(ok for _, ok in checks) else 1)
EOF

PY_RC=$?
if [ $PY_RC -eq 0 ]; then
    sed 's/^/  /' /tmp/test_embeddings_mm.out | sed "s/PASS/$(printf '%b' "${GREEN}PASS${NC}")/"
else
    cat /tmp/test_embeddings_mm.out
fi
FAILURES=$((FAILURES + PY_RC))

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}ALL PASS${NC} test_embeddings_mm"
    exit 0
else
    echo -e "${RED}${FAILURES} FAILURES${NC} test_embeddings_mm"
    exit 1
fi
