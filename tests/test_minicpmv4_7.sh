#!/bin/bash
# MiniCPM-V 4.7 (minicpmv4_7) live end-to-end on an MLX pack written by tests/convert_minicpmv4_7.py:
#
#   MINICPM_PACK=<pack dir> ./tests/test_minicpmv4_7.sh
#
#   [0] advertised: the Qwen3.5-MoE trunk, the vision capability, no video modality
#   [1] short answer; the stream and the non-stream answer are the same bytes
#   [2] a tool call round-trips through the template (arguments are valid JSON)
#   [3] images: the tower loads, house.jpeg is a thumbnail + 1x2 crops (196 soft tokens, the processor's
#       numbers), the canvas M-RoPE table is built, the subject is named, /v1/messages takes an image block
#   [4] a large image is SLICED (3x3 crops): small print and a footer far from the top are read exactly
#   [5] two images in one prompt are both seen, and a repeated image turn reuses the prefix
#   [6] --no-vision (second boot) serves text and refuses an image by name; a pack that ships no tower takes
#       the same path (hermetic: `a pack without the tower disables vision` in minicpm_vision.zig)
#
# Hermetic counterparts: minicpm_vision.zig (slicing, position ids), minicpm_canvas.zig and the server.zig
# `MiniCPM-V 4.7` test (expanded prompt + canvas positions == the processor and get_rope_index), and the
# parity test against the executed reference (tests/dump_minicpmv4_7_vision_fixtures.py, then
# MINICPM_VISION_MODEL=<pack> MINICPM_VISION_FIXTURE=<file> zig-out/tests/test).

set -euo pipefail

MODEL="${MINICPM_PACK:-}"
if [ -z "$MODEL" ]; then echo "SKIP: MINICPM_PACK not set"; exit 0; fi
if [ ! -f "$MODEL/config.json" ]; then echo "FAIL: $MODEL/config.json not found"; exit 1; fi
python3 -c "import PIL" 2>/dev/null || { echo "SKIP: PIL missing (needed to build the large-image fixture)"; exit 0; }

cd "$(dirname "$0")/.."
PORT="${MINICPM_TEST_PORT:-11371}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"
WORK=$(mktemp -d /tmp/minicpm_test.XXXXXX)
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

boot() { # boot <pack> <log>
    HOME="$WORK/home" "$BIN" --model "$1" ${NO_VISION:-} --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 > "$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        grep -q "Model ready (loaded on inference thread)" "$2" && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "FAIL: server died during load"; tail -20 "$2"; exit 1; }
        sleep 2
    done
    echo "FAIL: model did not load"; tail -20 "$2"; exit 1
}
stop() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }

pass=0; fail=0
check() { if grep -qF -- "$3" <<< "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; echo "  wanted: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); fi; }
check_i() { if grep -qiE -- "$3" <<< "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; echo "  wanted: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); fi; }
chat() { curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }
content() { python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")'; }
image_body() { # image_body <prompt> <path[,path...]>
    python3 - "$1" "$2" <<'PY'
import base64, json, sys
prompt, paths = sys.argv[1:3]
mime = lambda p: "image/png" if p.endswith(".png") else "image/jpeg"
parts = [{"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime(p), base64.b64encode(open(p, "rb").read()).decode())}} for p in paths.split(",")]
parts.append({"type": "text", "text": prompt})
print(json.dumps({"model": "m", "max_tokens": 120, "temperature": 0, "messages": [{"role": "user", "content": parts}],
                  "chat_template_kwargs": {"enable_thinking": False}}))
PY
}

LOG="$WORK/serve.log"
boot "$MODEL" "$LOG"

M=$(curl -s -m 30 "$BASE/v1/models")
check "[0] vision capability" "$M" '"vision"'
check "[0] image modality" "$M" '"image"'
if grep -qF '"video"' <<< "$M"; then echo "FAIL [0] video must not be advertised"; fail=$((fail+1)); else echo "PASS [0] no video modality"; pass=$((pass+1)); fi
check "[3] tower boot line" "$(cat "$LOG")" "MiniCPM-V 4.7 SigLIP2 ViT"

Q='{"model":"m","max_tokens":48,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]'
R1=$(chat "$Q}" | content)
check "[1] answers Paris" "$R1" "Paris"
S=$(curl -sN -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$Q,\"stream\":true}" | python3 -c '
import json,sys
out=""
for l in sys.stdin:
    if l.startswith("data: {"):
        c=json.loads(l[6:])["choices"]
        if c: out+=c[0]["delta"].get("content") or ""
print(out)')
[ "$S" = "$R1" ] && { echo "PASS [1] stream == non-stream"; pass=$((pass+1)); } || { echo "FAIL [1] stream differs: '$S' vs '$R1'"; fail=$((fail+1)); }

T=$(chat '{"model":"m","max_tokens":256,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"What is the weather in Beijing right now?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]}')
check "[2] tool call" "$T" '"name":"get_weather"'
python3 -c 'import json,sys; a=json.load(sys.stdin)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]; json.loads(a)' <<< "$T" 2>/dev/null \
    && { echo "PASS [2] arguments are JSON"; pass=$((pass+1)); } || { echo "FAIL [2] arguments are not JSON"; fail=$((fail+1)); }

HOUSE=tests/fixtures/house.jpeg
ROBOT=tests/fixtures/robot.png
A=$(chat "$(image_body "What is the main subject of this image? Answer in one word." "$HOUSE")" | content)
check_i "[3] house" "$A" "house|home|building"
LOGS=$(cat "$LOG")
# 487x730 -> a 28x40 thumbnail and two 36x28 crops: (7*10 + 2*9*7) = 196 soft tokens in the processor.
check "[3] sliced into a thumbnail and 1x2 crops" "$LOGS" "Decoded 730x487 image → minicpm 3 piece(s), slice grid 1x2"
check "[3] soft tokens equal the processor's" "$LOGS" "196 image + 0 video"
check "[3] canvas M-RoPE built" "$LOGS" "Canvas M-RoPE: 3 crops"
AN=$(curl -s -m 300 "$BASE/v1/messages" -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' --data-binary "$(python3 - "$HOUSE" <<'PY'
import base64, json, sys
d = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({"model": "m", "max_tokens": 40, "messages": [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}},
    {"type": "text", "text": "What is the main subject? One word."}]}]}))
PY
)")
check_i "[3] /v1/messages image block" "$AN" "house|home|building"

python3 - "$WORK/text.png" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (1800, 1300), "white")
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
except Exception:
    f = big = ImageFont.load_default()
d.text((60, 60), "ACCESS CODE: BLUE-4721", fill="black", font=big)
for i in range(14):
    d.text((60, 220 + i * 60), "lorem ipsum dolor sit amet consectetur %d" % i, fill="gray", font=f)
d.text((1100, 1180), "footer ref 90210", fill="black", font=f)
img.save(sys.argv[1])
PY
T4=$(chat "$(image_body "Read the access code and the footer reference exactly." "$WORK/text.png")" | content)
check "[4] access code read" "$T4" "BLUE-4721"
check "[4] footer read" "$T4" "90210"
check "[4] sliced into 3x3 crops" "$(cat "$LOG")" "slice grid 3x3"

TWO=$(chat "$(image_body "Describe both images in one short sentence each." "$HOUSE,$ROBOT")" | content)
check_i "[5] first image seen" "$TWO" "house|home|building"
check_i "[5] second image seen" "$TWO" "robot|android|cyborg"
chat "$(image_body "What is the main subject of this image? Answer in one word." "$HOUSE")" > "$WORK/again.json"
CACHED=$(python3 -c 'import json; print(json.load(open("'"$WORK"'/again.json"))["usage"]["prompt_tokens_details"]["cached_tokens"])')
[ "$CACHED" -gt 0 ] && { echo "PASS [5] repeated image turn reuses the prefix ($CACHED tokens)"; pass=$((pass+1)); } || { echo "FAIL [5] no prefix reuse"; fail=$((fail+1)); }
check "[5] no MLX error" "$( (grep -c '\[mlx\] error' "$LOG" || true) )" "0"
stop

# [6] --no-vision drops the tower at load: text serves, an image is refused by name.
LOG2="$WORK/serve2.log"
NO_VISION=--no-vision boot "$MODEL" "$LOG2"
R6=$(chat "$Q}" | content)
check "[6] --no-vision answers" "$R6" "Paris"
M6=$(curl -s -m 30 "$BASE/v1/models")
if grep -qF '"vision"' <<< "$M6"; then echo "FAIL [6] vision advertised without a tower"; fail=$((fail+1)); else echo "PASS [6] vision not advertised"; pass=$((pass+1)); fi
I6=$(chat "$(image_body "What is this?" "$HOUSE")")
check_i "[6] image refused by name" "$I6" "vision|image"
echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
