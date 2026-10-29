#!/usr/bin/env bash
# test_qwen_video_input.sh — Qwen video input: four DISTINCT frames as one `video_url` (house, robot, street signs, an
# app screenshot). At least two frames' subjects must be named AND something only the PIXELS supply (frame 3's sign
# text): a wiring bug that threads frame 0 into every temporal-patch group, or encodes frames and never splices them,
# still answers plausibly about one subject. On a checkpoint that cannot read the signs this is a checkpoint
# expectation, not a wire failure (qwen4_exp reads them).
#
#   tests/test_qwen_video_input.sh [model_dir] [port]

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
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

srv_wait_health "localhost:$PORT" 90 1 || true
if ! curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then echo "FAIL: server never came up"; cat "$LOG"; exit 1; fi

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
# Something only the PIXELS supply: the generic subjects above passed for weeks
# on a build that encoded the frames and never spliced them (the model narrated
# a plausible clip). Frame 3's signs read "Grey Fox Trl" / "Waterfall Dr" / "STOP".
if echo "$TEXT" | grep -qiE "stop|gr[ae]y fox|waterfall"; then echo "  OK: sign text from frame 3 surfaced (pixel-only content)"; else echo "  FAIL: no pixel-only content (STOP / Grey Fox / Waterfall) — frames may not have reached the prompt"; FAIL=1; fi

echo "== Video decode + M-RoPE engagement (server log) =="
grep -qE "Decoded 4 frames → qwen video" "$LOG" && echo "  OK: video decode ran (4 real frames)" || { echo "  FAIL: video decode did not engage"; FAIL=1; }
grep -qE "M-RoPE: 0 images, 1 videos" "$LOG" && echo "  OK: M-RoPE engaged on the video block" || { echo "  FAIL: M-RoPE did not engage for video"; FAIL=1; }

if [ "$FAIL" = "0" ]; then echo "PASS: qwen video input e2e"; else echo "FAIL: qwen video input e2e"; cat "$LOG" | tail -30; fi
exit $FAIL
