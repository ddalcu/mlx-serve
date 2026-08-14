#!/usr/bin/env bash
# LFM2-VL image input end-to-end over HTTP (OpenAI + Anthropic surfaces).
#
# The two things output alone cannot see are asserted from the server's own log:
# the SPLIT DECISION (one resized image vs a tile grid + thumbnail) and the token
# accounting the splice depends on. A tiled image that silently fell back to one
# low-res view still answers plausibly — it just cannot read anything small,
# which is the whole reason the tiling exists.
#
# Usage: tests/test_lfm2_vision.sh [model_dir] [port]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${LFM2_VISION_MODEL:-/Volumes/G Drive SSD/models-dl/LiquidAI/LFM2.5-VL-3B-MLX-4bit}}"
PORT="${2:-11388}"
HOUSE="tests/fixtures/house.jpeg"
ROBOT="tests/fixtures/robot.png"

if [ ! -f "$MODEL/config.json" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
python3 -c "import PIL" 2>/dev/null || { echo "SKIP: PIL missing (needed to build the fine-print fixture)"; exit 0; }

WORK=$(mktemp -d); LOG="$WORK/server.log"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT

for i in $(seq 1 240); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -s "localhost:$PORT/health" >/dev/null 2>&1; then echo "FAIL: server never came up"; tail -20 "$LOG"; exit 1; fi

FAIL=0
ask() { # ask <base64> <prompt> <max_tokens> <mime>
  cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @- |
{"model":"lfm2","max_tokens":$3,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"$2"},
 {"type":"image_url","image_url":{"url":"data:$4;base64,$1"}}]}]}
EOF
  python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null
}

echo "== [1/7] the tower loads and vision is advertised =="
grep -q "LFM2-VL SigLIP2-NaFlex ViT" "$LOG" \
  && echo "  OK: encoder boot line" || { echo "  FAIL: no LFM2-VL encoder line (wrong tower or none)"; FAIL=1; }
# Resolved by STRING from the tokenizer, not config.json — a checkpoint that
# renames them would silently splice an unlabelled run of pads.
grep -qE "LFM2-VL image tokens: <image>=[1-9][0-9]* start=[1-9][0-9]* end=[1-9][0-9]* thumbnail=[1-9][0-9]* row_col_base=[1-9][0-9]*" "$LOG" \
  && echo "  OK: image marker ids resolved" || { echo "  FAIL: image marker ids unresolved"; FAIL=1; }
curl -s "localhost:$PORT/v1/models" | grep -q '"vision"' \
  && echo "  OK: vision capability" || { echo "  FAIL: vision capability not advertised"; FAIL=1; }

echo "== [2/7] single-tile image (under the budget) =="
B_HOUSE=$(base64 -i "$HOUSE" | tr -d '\n')
T=$(ask "$B_HOUSE" "What is the main subject of this image? One word." 32 "image/jpeg")
echo "  -> $T"
echo "$T" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house'"; FAIL=1; }
# 730x487 sits under max_image_tokens x tolerance, so the reference resizes it
# whole rather than tiling — and lands on a 26x38 grid, 247 tokens.
grep -q "lfm2 grid 26x38 (247 tokens, resized 608x416)" "$LOG" \
  && echo "  OK: single-tile geometry matches the reference" \
  || { echo "  FAIL: wrong single-tile geometry"; grep "Decoded" "$LOG" | tail -2; FAIL=1; }

echo "== [3/7] a second arch-shaped image still answers =="
B_ROBOT=$(base64 -i "$ROBOT" | tr -d '\n')
T=$(ask "$B_ROBOT" "What is in this image? One word." 32 "image/png")
echo "  -> $T"
echo "$T" | grep -qiE "robot|android|cyborg|humanoid" || { echo "  FAIL: expected 'robot'"; FAIL=1; }

echo "== [4/7] a large image SPLITS into tiles + thumbnail =="
python3 - "$WORK" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont
work = sys.argv[1]
img = Image.new("RGB", (1800, 1400), "white")
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
except Exception:
    f = ImageFont.load_default()
for i, t in enumerate(["Serial: QX-88231-KLM", "Batch code: 7741-ZZ", "Checksum: 0xBEEF42"]):
    d.text((1180, 1180 + i * 22), t, fill="black", font=f)
d.ellipse([200, 200, 900, 900], outline="purple", width=10)
img.save(work + "/fine.png")
# The same picture at the resolution the UNTILED path would have used.
img.resize((576, 416), Image.BICUBIC).save(work + "/fine_small.png")
PY
B_FINE=$(base64 -i "$WORK/fine.png" | tr -d '\n')
FINE=$(ask "$B_FINE" "List every line of small text in the image, exactly as written." 120 "image/png")
echo "  -> $(echo "$FINE" | tr '\n' '|')"
# 1800x1400 -> a 2x3 tile grid on a 1536x1024 canvas + a 252-token thumbnail
# (448x576): 6*256 + 252 = 1788, the reference's own numbers.
grep -q "lfm2 2x3 tiles + thumbnail on a 1536x1024 canvas (1788 tokens)" "$LOG" \
  && echo "  OK: tile grid + thumbnail match the reference" \
  || { echo "  FAIL: wrong tile layout"; grep "Decoded" "$LOG" | tail -2; FAIL=1; }
grep -q "Multimodal: processing 7 image(s)" "$LOG" \
  && echo "  OK: 6 tiles + thumbnail each encoded separately" \
  || { echo "  FAIL: tiles were not encoded as separate images"; grep "Multimodal: processing" "$LOG" | tail -2; FAIL=1; }
grep -q "Inserted 1788 image" "$LOG" \
  && echo "  OK: every tile's pads reached the prompt" \
  || { echo "  FAIL: pad count does not match the encoder output"; grep "Inserted" "$LOG" | tail -2; FAIL=1; }

echo "== [5/7] tiling is what makes the fine print legible =="
# The bar is a comparison, not an absolute: this checkpoint reads large text
# fine at thumbnail resolution, so only a side-by-side shows the tiles carrying
# detail. Same prompt, same picture, one downscaled to the untiled geometry.
HITS=0
for want in "QX-88231-KLM" "7741-ZZ" "0xBEEF42"; do
  echo "$FINE" | grep -q "$want" && HITS=$((HITS+1))
done
B_SMALL=$(base64 -i "$WORK/fine_small.png" | tr -d '\n')
SMALL=$(ask "$B_SMALL" "List every line of small text in the image, exactly as written." 120 "image/png")
SMALL_HITS=0
for want in "QX-88231-KLM" "7741-ZZ" "0xBEEF42"; do
  echo "$SMALL" | grep -q "$want" && SMALL_HITS=$((SMALL_HITS+1))
done
echo "  tiled: $HITS/3 exact, downscaled: $SMALL_HITS/3"
[ "$HITS" -eq 3 ] || { echo "  FAIL: tiled read should be exact"; FAIL=1; }
[ "$SMALL_HITS" -lt "$HITS" ] || echo "  NOTE: the downscaled arm read it too — this fixture no longer discriminates"

echo "== [6/7] thinking-on STREAMING puts the answer in content, not reasoning =="
# A stream starts inside a think block only when the RENDERED PROMPT ends inside
# one. Seeding it from the request flag instead routed LFM2-VL's whole answer
# into reasoning_content and left content EMPTY — the app drew a Thinking block
# with no reply under it (live 2026-08-13). LFM2-VL is the checkpoint that shows
# it (its generation prompt is a bare `<|im_start|>assistant`), but the class is
# every model whose template does not render the opener. The bar is the
# stream-vs-non-stream invariant, not a phrasing check.
think_req() { # think_req <stream>
  cat <<EOF | curl -s -N --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-
{"model":"lfm2","enable_thinking":true,"stream":$1,"max_tokens":80,"temperature":0,
 "messages":[{"role":"user","content":"Name three primary colors."}]}
EOF
}
STREAM_OUT=$(think_req true | python3 -c "
import json,sys
c=r=''
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data: '): continue
    b=line[6:]
    if b=='[DONE]': break
    try: d=json.loads(b)
    except: continue
    ch=d.get('choices') or []
    if not ch: continue
    de=ch[0].get('delta') or {}
    c+=de.get('content') or ''; r+=de.get('reasoning_content') or ''
print(json.dumps({'content':c,'reasoning':r}))
")
NON_OUT=$(think_req false | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
print(json.dumps({'content':m.get('content') or '','reasoning':m.get('reasoning_content') or ''}))
")
S_C=$(echo "$STREAM_OUT" | python3 -c "import json,sys;print(json.load(sys.stdin)['content'])")
N_C=$(echo "$NON_OUT" | python3 -c "import json,sys;print(json.load(sys.stdin)['content'])")
echo "  stream content   -> $(echo "$S_C" | head -c 60)"
[ -n "$S_C" ] || { echo "  FAIL: streaming content is EMPTY (the whole answer went to reasoning_content)"; FAIL=1; }
[ "$S_C" = "$N_C" ] && echo "  OK: streaming and non-streaming agree byte for byte" \
  || { echo "  FAIL: stream/non-stream disagree"; echo "    non-stream -> $(echo "$N_C" | head -c 60)"; FAIL=1; }

echo "== [7/7] Anthropic /v1/messages carries images too =="
ANT=$(cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/messages" -H 'content-type: application/json' -d @- |
{"model":"lfm2","max_tokens":32,"messages":[{"role":"user","content":[
 {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"$B_HOUSE"}},
 {"type":"text","text":"What is the main subject? One word."}]}]}
EOF
python3 -c "import sys,json;d=json.load(sys.stdin);print(''.join(b.get('text','') for b in d.get('content',[])))" 2>/dev/null)
echo "  -> $ANT"
echo "$ANT" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house' on /v1/messages"; FAIL=1; }

echo
if [ "$FAIL" -eq 0 ]; then echo "PASS: LFM2-VL vision"; else echo "FAIL: LFM2-VL vision"; fi
exit "$FAIL"
