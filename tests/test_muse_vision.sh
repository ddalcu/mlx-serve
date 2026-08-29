#!/usr/bin/env bash
# Muse-Glimmer image input end-to-end over HTTP (OpenAI + Anthropic surfaces).
# Asserts a relevant answer, the token accounting the splice depends on, two
# images in one turn, and that --no-vision drops the tower. Skips cleanly when
# the model is absent so CI stays green.
#
# Usage: tests/test_muse_vision.sh [model_dir] [port]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${MUSE_VISION_MODEL:-$HOME/.mlx-serve/models/ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit}}"
PORT="${2:-11386}"
HOUSE="tests/fixtures/house.jpeg"
ROBOT="tests/fixtures/robot.png"

if [ ! -f "$MODEL/config.json" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi

LOG=$(mktemp); NVLOG=$(mktemp)
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 240); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -s "localhost:$PORT/health" >/dev/null 2>&1; then echo "FAIL: server never came up"; tail -20 "$LOG"; exit 1; fi

B_HOUSE=$(base64 -i "$HOUSE"); B_ROBOT=$(base64 -i "$ROBOT")
FAIL=0

echo "== /v1/models advertises vision =="
curl -s "localhost:$PORT/v1/models" | grep -q '"vision"' \
  && echo "  OK" || { echo "  FAIL: vision capability not advertised"; FAIL=1; }

echo "== OpenAI /v1/chat/completions =="
OAI=$(cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-
{"model":"muse","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the main subject of this image? One word."},
 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$B_HOUSE"}}]}]}
EOF
)
OAI_TEXT=$(echo "$OAI" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  -> $OAI_TEXT"
echo "$OAI_TEXT" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house'"; FAIL=1; }
if [ -f "$MODEL/drafter/config.json" ]; then
  grep -qE "\[spec-stats\] mode=dflash attempts=[1-9][0-9]*" "$LOG" \
    && echo "  OK: Muse vision kept DFlash engaged" \
    || { echo "  FAIL: Muse vision silently disabled DFlash"; grep -E "spec-wiring|spec-stats" "$LOG" | tail -5; FAIL=1; }
fi

echo "== token accounting (the splice depends on it) =="
# 730x487 → smart_resize 756x504 → grid 54x36 patches → (54/2)*(36/2) = 486 pads,
# plus <|image_start|>/<|image_end|>. A drift here scatters the tower's rows
# into the wrong positions rather than failing loudly.
# The arch name comes from `@tagName(vp.mode)` (server.zig), i.e. a Zig enum
# tag, so it is always lowercase — an older literal "Muse" in this pattern
# broke the assertion on a refactor while the math stayed correct. Match
# case-insensitively; the NUMBERS are the contract.
grep -qiE "muse grid 36x54 \(486 tokens, resized 756x504\)" "$LOG" \
  && echo "  OK: grid 36x54 → 486 tokens" || { echo "  FAIL: unexpected grid/token math"; grep -E "Decoded .* image" "$LOG"; FAIL=1; }
grep -qE "Inserted 486 image .* \(prompt: [0-9]+ -> [0-9]+ tokens\)" "$LOG" \
  && echo "  OK: 486 pads spliced" || { echo "  FAIL: pad run not inserted"; FAIL=1; }

echo "== two images in one turn =="
TWO=$(cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-
{"model":"muse","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"Two images follow. Name the main subject of each, in order, one word each."},
 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$B_HOUSE"}},
 {"type":"image_url","image_url":{"url":"data:image/png;base64,$B_ROBOT"}}]}]}
EOF
)
TWO_TEXT=$(echo "$TWO" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  -> $(echo "$TWO_TEXT" | tr '\n' ' ')"
echo "$TWO_TEXT" | grep -qi "house" && echo "$TWO_TEXT" | grep -qiE "robot|android|cyborg" \
  || { echo "  FAIL: expected both subjects"; FAIL=1; }

echo "== Anthropic /v1/messages =="
ANT=$(cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/messages" -H 'content-type: application/json' -d @-
{"model":"muse","max_tokens":64,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the main subject of this image? One word."},
 {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"$B_HOUSE"}}]}]}
EOF
)
ANT_TEXT=$(echo "$ANT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(''.join(b.get('text','') for b in d.get('content',[])))" 2>/dev/null)
echo "  -> $ANT_TEXT"
echo "$ANT_TEXT" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house'"; FAIL=1; }

kill "$SRV" 2>/dev/null; sleep 2

echo "== --no-vision drops the tower =="
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --no-vision --log-level info > "$NVLOG" 2>&1 &
SRV=$!
for i in $(seq 1 240); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if grep -q "Muse-Glimmer ViT" "$NVLOG"; then echo "  FAIL: tower loaded under --no-vision"; FAIL=1; else echo "  OK: tower not loaded"; fi
curl -s "localhost:$PORT/v1/models" | grep -q '"vision"' \
  && { echo "  FAIL: vision still advertised"; FAIL=1; } || echo "  OK: vision not advertised"

if [ "$FAIL" = "0" ]; then echo "PASS: muse vision e2e"; else echo "FAIL: muse vision e2e"; tail -20 "$LOG"; fi
exit $FAIL
