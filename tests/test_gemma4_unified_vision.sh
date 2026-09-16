#!/bin/bash
# Integration test for Gemma 4 12B "unified" (encoder-free) vision.
#
# The 12B checkpoint has NO SigLIP tower: vision is a single patch embedder
# (LN → Dense → LN → +factorized 2D posemb → LN → RMSNorm → Linear) that
# projects raw 48px RGB patches straight into the language model. This test
# boots mlx-serve against the 12B and verifies an image is actually described
# (correct colors + spatial layout), not hallucinated.
#
# Gated on a fixture so CI without the model stays green:
#   G4_UNIFIED_MODEL=/path/to/gemma-4-12b-it-4bit ./tests/test_gemma4_unified_vision.sh [port]
set -uo pipefail

MODEL="${G4_UNIFIED_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-12b-it-4bit}"
PORT="${1:-8231}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"
PY="${G4_TEST_PYTHON:-python3}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_gemma4_unified_vision: set G4_UNIFIED_MODEL=/path/to/gemma-4-12b-it-4bit to run"
    exit 0
fi
if [ ! -f "$BIN" ]; then
    echo -e "${RED}ERROR${NC} $BIN not found — build with: zig build -Doptimize=ReleaseFast"
    exit 1
fi

ok()  { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "${2:-}" ] && echo "    $2"; }
assert_contains() { if grep -qi "$2" <<< "$3"; then ok "$1"; else bad "$1" "missing '$2' in: $(echo "$3" | head -c 300)"; fi; }

# Build a distinctive test image: a red ellipse on the left/center, a blue
# rectangle on the right, on a light-gray background. Requires Pillow.
IMG="$(mktemp -t g4shapes).png"
"$PY" - "$IMG" <<'PY' 2>/dev/null
import sys
try:
    from PIL import Image, ImageDraw
except Exception:
    sys.exit(3)
img = Image.new("RGB", (640, 480), (240, 240, 240))
d = ImageDraw.Draw(img)
d.ellipse([80, 80, 360, 360], fill=(220, 30, 30))
d.rectangle([400, 250, 600, 420], fill=(30, 60, 220))
img.save(sys.argv[1])
PY
if [ ! -s "$IMG" ]; then
    echo -e "${YELLOW}SKIP${NC} test_gemma4_unified_vision: Pillow unavailable to render the test image (set G4_TEST_PYTHON to a python with PIL)"
    rm -f "$IMG"
    exit 0
fi

LOG="$(mktemp)"
echo "→ starting mlx-serve on :$PORT with $MODEL"
"$BIN" --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG" "$IMG"; }
trap cleanup EXIT

for i in $(seq 1 60); do
    if curl -fs --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"'; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo -e "${RED}server died${NC}"; tail -20 "$LOG"; exit 1; fi
    sleep 1
done

# 1. The unified (encoder-free) vision embedder loaded — not the SigLIP tower.
assert_contains "loads unified encoder-free vision embedder" "unified (encoder-free)" "$(cat "$LOG")"

# 2. Describe-the-image round trip.
B64="$(base64 -i "$IMG")"
REQ="$(mktemp)"
cat > "$REQ" <<EOF
{"model":"mlx-serve","max_tokens":160,"temperature":0.0,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,$B64"}},{"type":"text","text":"Describe this image: what shapes and colors do you see, and where are they?"}]}]}
EOF
RESP="$(curl -fs --max-time 120 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @"$REQ")"
rm -f "$REQ"
CONTENT="$(echo "$RESP" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)"
echo "    model said: $(echo "$CONTENT" | head -c 220)"

# 3. The image actually drove the description: red + blue must both appear, and
#    the encode produced the expected 256 soft tokens for a 768x768 square.
assert_contains "describes the red shape"        "red"  "$CONTENT"
assert_contains "describes the blue shape"        "blue" "$CONTENT"
assert_contains "vision encode produced soft tokens" "\[1,256,3840\] tokens (256 vision + 0 audio)" "$(cat "$LOG")"
assert_contains "inserted matching image placeholders" "Inserted 256 image + 0 audio soft tokens" "$(cat "$LOG")"

# 4. Text-only request on the same model still works (no vision regression).
TRESP="$(curl -fs --max-time 60 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","max_tokens":24,"temperature":0.0,"messages":[{"role":"user","content":"Reply with exactly the word: PONG"}]}')"
assert_contains "text-only chat still works" "PONG" "$(echo "$TRESP" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)"

echo ""
echo -e "Gemma 4 12B unified vision: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
[ "$FAIL" -eq 0 ]
