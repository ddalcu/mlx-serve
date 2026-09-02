#!/bin/bash
# Guard: SeedVR2 restoration answers with a PICTURE, or refuses BY NAME.
#
# The failure this exists for is a 200 with nothing wrong-looking about it.
# MLX at the Metal working-set edge returns degenerate values instead of
# raising, so a restore that does not fit in memory comes back as a full-size
# PNG of ONE FLAT COLOUR — pure white in the report that started this
# (1808x1920, every pixel 255), a flat grey the run before, a brown and then a
# green rectangle at 1536x1536 in the same session. The colour changes run to
# run, which is the tell. Bigger still, the dense mid-block attention asked for
# 17 GB in one buffer and killed the process outright.
#
# So OUTPUT EQUALITY IS NOT THE BAR HERE — a wrong-but-plausible restoration is
# a model question. The bar is: a size that fits comes back with VARIANCE in
# it, a size that does not is a 400 that names the numbers it compared, and the
# server is still answering afterwards.
#
# Needs the 3B checkpoint (~6.8 GB); SKIPS cleanly without it.
#
# Usage: ./tests/test_seedvr2_upscale.sh [port]

set -u

PORT="${1:-11291}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

MODEL_DIR=""
for root in "$HOME/.mlx-serve/models"/*/*SeedVR2*; do
    [ -f "$root/dit.safetensors" ] && MODEL_DIR="$root" && break
done
if [ -z "$MODEL_DIR" ]; then
    echo -e "${YELLOW}[skip]${NC} no SeedVR2 checkpoint under ~/.mlx-serve/models"
    exit 0
fi
MODEL_ID="$(basename "$(dirname "$MODEL_DIR")")/$(basename "$MODEL_DIR")"
echo "[info] model $MODEL_ID"

PY="${PY:-python3}"
if ! "$PY" -c "import PIL, numpy" 2>/dev/null; then
    echo -e "${YELLOW}[skip]${NC} needs python with pillow + numpy (set PY=/path/to/python)"
    exit 0
fi

TMP="$(mktemp -d)"
cleanup() {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

# A synthetic source with real structure in it: a flat input could not tell a
# working restore from a broken one, since both come back flat.
"$PY" - "$TMP" <<'PYEOF'
import sys, numpy as np
from PIL import Image
d = sys.argv[1]
y, x = np.mgrid[0:256, 0:256]
img = np.stack([
    ((x // 16 + y // 16) % 2) * 200 + 30,
    (x * 255 // 255),
    (y * 255 // 255),
], -1).astype(np.uint8)
Image.fromarray(img).save(f"{d}/src.png")
PYEOF

"$BINARY" serve --port "$PORT" --host 127.0.0.1 --skip-mem-preflight > "$TMP/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 60); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    sleep 1
done

post() {  # $1 = request json path, $2 = output path -> echoes status code
    curl -s -X POST "http://127.0.0.1:$PORT/v1/images/upscales" \
        -H 'Content-Type: application/json' -d @"$1" -o "$2" -w "%{http_code}" --max-time 3600
}

# ── [1] a size that fits comes back as a picture ─────────────────────────
"$PY" - "$TMP" "$MODEL_ID" <<'PYEOF'
import sys, json, base64
d, mid = sys.argv[1], sys.argv[2]
json.dump({"model": mid, "image": base64.b64encode(open(f"{d}/src.png", "rb").read()).decode(), "seed": 7},
          open(f"{d}/req.json", "w"))
PYEOF
CODE="$(post "$TMP/req.json" "$TMP/out.json")"
check "256x256 restore answers 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"

if [ "$CODE" = "200" ]; then
    VERDICT="$("$PY" - "$TMP" <<'PYEOF'
import sys, json, base64, io
import numpy as np
from PIL import Image
d = sys.argv[1]
png = base64.b64decode(json.load(open(f"{d}/out.json"))["data"][0]["b64_json"])
a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)
# A degenerate frame has essentially no variance; a real restoration of this
# source has a lot. The threshold sits far from both (measured std ~1.2 on the
# grey failure, 0.0 on the white one, 90+ on every good restore).
print(f"{a.shape[0]}x{a.shape[1]} std={a.std():.2f}")
PYEOF
)"
    echo "  [info] output $VERDICT"
    STD="${VERDICT##*std=}"
    check "output is not a flat rectangle (std $STD)" \
        "$(awk -v s="$STD" 'BEGIN{print (s > 20) ? 1 : 0}')"
    check "output keeps the source geometry" \
        "$(echo "$VERDICT" | grep -q '^256x256 ' && echo 1 || echo 0)"
fi

# ── [2] a size that cannot fit is refused BY NAME, not ruined ────────────
# 8000x8000 needs ~350 GB of transient — past any Mac, so this arm does not
# depend on how much memory the runner happens to have free.
"$PY" - "$TMP" "$MODEL_ID" <<'PYEOF'
import sys, json, base64, io
from PIL import Image
d, mid = sys.argv[1], sys.argv[2]
buf = io.BytesIO()
Image.new("RGB", (8000, 8000), (128, 64, 32)).save(buf, format="PNG")
json.dump({"model": mid, "image": base64.b64encode(buf.getvalue()).decode()},
          open(f"{d}/big.json", "w"))
PYEOF
CODE="$(post "$TMP/big.json" "$TMP/big_out.json")"
check "an unfittable size is refused (got $CODE, want 400)" "$([ "$CODE" = "400" ] && echo 1 || echo 0)"
# The refusal quotes what it compared, or it reads as "this should have worked".
check "the refusal names the memory it needed" \
    "$(grep -q 'GB of free memory' "$TMP/big_out.json" && echo 1 || echo 0)"
check "the refusal names the geometry" \
    "$(grep -q '8000x8000' "$TMP/big_out.json" && echo 1 || echo 0)"

# ── [3] the refusal is a refusal, not a crash ────────────────────────────
check "server still answering after the refusal" \
    "$(curl -s -m 5 "http://127.0.0.1:$PORT/health" > /dev/null && echo 1 || echo 0)"

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
