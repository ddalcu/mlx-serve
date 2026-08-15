#!/usr/bin/env bash
# Qwen3-VL video input end-to-end over HTTP: a `video_url` content block
# carrying multiple DISTINCT frame images (house / robot / street signs / a
# "not hot dog" app screenshot — no real video clip is checked into fixtures/,
# so genuinely different still images stand in for genuinely different
# frames). This is the class guard for the video wire path: a wiring bug that
# threads only frame 0 into every temporal-patch group (the exact regression
# `qwen_vision.zig`'s "buildPixelValuesVideo reads REAL per-frame data, not
# one frame duplicated" unit test guards in isolation) would still answer
# plausibly here — it would just describe ONE subject instead of several, so
# the model is asked to enumerate what it saw across frames rather than
# describe "an image".
#
# Usage: tests/test_qwen_video_input.sh [model_dir] [port]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${QWEN_VISION_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}}"
PORT="${2:-11386}"
F1="tests/fixtures/house.jpeg"
F2="tests/fixtures/robot.png"
F3="tests/fixtures/street-name-signs.jpg"
F4="tests/fixtures/not_hot_dog_app.webp"

if [ ! -f "$MODEL/config.json" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
for f in "$F1" "$F2" "$F3" "$F4"; do
  if [ ! -f "$f" ]; then echo "SKIP: fixture $f missing"; exit 0; fi
done

LOG=$(mktemp)
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 90); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -s "localhost:$PORT/health" >/dev/null 2>&1; then echo "FAIL: server never came up"; cat "$LOG"; exit 1; fi

mime_for() {
  case "$1" in
    *.jpeg|*.jpg) echo "image/jpeg" ;;
    *.png) echo "image/png" ;;
    *.webp) echo "image/webp" ;;
  esac
}
frame_json() {
  local f="$1" mime; mime=$(mime_for "$f")
  printf '"data:%s;base64,%s"' "$mime" "$(base64 -i "$f")"
}
FAIL=0

echo "== /v1/models advertises video capability =="
MODELS=$(curl -s "localhost:$PORT/v1/models")
echo "$MODELS" | grep -q '"video"' && echo "  OK: video in input_modalities" || { echo "  FAIL: no video capability advertised"; FAIL=1; }

echo "== OpenAI /v1/chat/completions with a video_url block (4 distinct frames) =="
BODY=$(cat <<EOF
{"model":"qwen","max_tokens":128,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"Each frame of this video shows something different. List, in one short sentence per item, every distinct subject you can identify across all the frames."},
 {"type":"video_url","video_url":{"frames":[$(frame_json "$F1"),$(frame_json "$F2"),$(frame_json "$F3"),$(frame_json "$F4")]}}]}]}
EOF
)
RESP=$(echo "$BODY" | curl -s "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-)
TEXT=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  -> $TEXT"
if [ -z "$TEXT" ]; then echo "  FAIL: no completion text (request likely errored)"; echo "$RESP" | head -c 500; FAIL=1; fi

# The class guard: a pipeline that collapsed all frames to one would still
# answer plausibly but would only ever mention ONE of these subjects. At
# least two of the four distinct fixtures should surface in the answer.
HITS=0
echo "$TEXT" | grep -qiE "house|home|building" && HITS=$((HITS + 1))
echo "$TEXT" | grep -qiE "robot" && HITS=$((HITS + 1))
echo "$TEXT" | grep -qiE "street|sign|road" && HITS=$((HITS + 1))
echo "$TEXT" | grep -qiE "hot ?dog|app|screenshot|phone" && HITS=$((HITS + 1))
echo "  matched $HITS/4 distinct-frame subjects"
if [ "$HITS" -lt 2 ]; then echo "  FAIL: fewer than 2 distinct subjects recognized (frames may have collapsed to one)"; FAIL=1; fi

echo "== Video decode + M-RoPE engagement (server log) =="
grep -qE "Decoded 4 frames → qwen video" "$LOG" && echo "  OK: video decode ran (4 real frames)" || { echo "  FAIL: video decode did not engage"; FAIL=1; }
grep -qE "M-RoPE: 0 images, 1 videos" "$LOG" && echo "  OK: M-RoPE engaged on the video block" || { echo "  FAIL: M-RoPE did not engage for video"; FAIL=1; }

if [ "$FAIL" = "0" ]; then echo "PASS: qwen video input e2e"; else echo "FAIL: qwen video input e2e"; cat "$LOG" | tail -30; fi
exit $FAIL
