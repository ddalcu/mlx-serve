#!/bin/bash
# Integration test for Gemma 4 12B "unified" (encoder-free) audio.
#
# The 12B checkpoint has NO conformer audio encoder: raw 16 kHz mono waveform is
# chunked into 640-sample frames (40 ms/token) and projected straight into the
# language model (RMSNorm → Linear). This test synthesizes a known phrase with
# macOS `say`, feeds it as an OpenAI `input_audio` block (raw float32-LE PCM),
# and verifies the model recognizes the spoken words — proving the audio splice
# path is live, not hallucinated.
#
# Gated on the model fixture + macOS tooling (say + ffmpeg/afconvert):
#   G4_UNIFIED_MODEL=/path/to/gemma-4-12b-it-4bit ./tests/test_gemma4_unified_audio.sh [port]
set -uo pipefail

MODEL="${G4_UNIFIED_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-12b-it-4bit}"
PORT="${1:-8231}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"
PY="${G4_TEST_PYTHON:-python3}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_gemma4_unified_audio: set G4_UNIFIED_MODEL=/path/to/gemma-4-12b-it-4bit to run"; exit 0
fi
if ! command -v say >/dev/null 2>&1; then
    echo -e "${YELLOW}SKIP${NC} test_gemma4_unified_audio: macOS \`say\` unavailable"; exit 0
fi
if [ ! -f "$BIN" ]; then
    echo -e "${RED}ERROR${NC} $BIN not found — build with: zig build -Doptimize=ReleaseFast"; exit 1
fi

ok()  { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "${2:-}" ] && echo "    $2"; }
assert_contains() { if grep -qi "$2" <<< "$3"; then ok "$1"; else bad "$1" "missing '$2' in: $(echo "$3" | head -c 300)"; fi; }

# Synthesize a distinctive phrase → raw float32-LE 16 kHz mono PCM.
PHRASE="the quick brown fox jumps over the lazy dog"
AIFF="$(mktemp -t g4say).aiff"; PCM="$(mktemp -t g4pcm).f32"
say -o "$AIFF" "$PHRASE" 2>/dev/null
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -i "$AIFF" -ar 16000 -ac 1 -f f32le "$PCM" >/dev/null 2>&1
else
    WAV="$(mktemp -t g4wav).wav"
    afconvert -f WAVE -d LEF32@16000 -c 1 "$AIFF" "$WAV" >/dev/null 2>&1
    # Strip the WAV header to the raw 'data' chunk payload.
    "$PY" - "$WAV" "$PCM" <<'PY' 2>/dev/null
import sys, struct
data = open(sys.argv[1], "rb").read()
i = data.find(b"data")
payload = data[i+8:] if i >= 0 else b""
open(sys.argv[2], "wb").write(payload)
PY
    rm -f "$WAV"
fi
if [ ! -s "$PCM" ]; then
    echo -e "${YELLOW}SKIP${NC} test_gemma4_unified_audio: could not produce raw PCM (need ffmpeg or afconvert)"
    rm -f "$AIFF" "$PCM"; exit 0
fi

LOG="$(mktemp)"
echo "→ starting mlx-serve on :$PORT with $MODEL"
"$BIN" --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG" "$AIFF" "$PCM"; }
trap cleanup EXIT
for i in $(seq 1 60); do
    if curl -fs --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"'; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo -e "${RED}server died${NC}"; tail -20 "$LOG"; exit 1; fi
    sleep 1
done

# The model must advertise the `audio` capability so the app can gate the mic.
MODELS="$(curl -fs --max-time 5 "$BASE/v1/models")"
assert_contains "/v1/models advertises audio capability" '"audio"' "$MODELS"

B64="$(base64 -i "$PCM")"
REQ="$(mktemp)"
cat > "$REQ" <<EOF
{"model":"mlx-serve","max_tokens":120,"temperature":0.0,"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"$B64","format":"mlx_pcm_f32"}},{"type":"text","text":"You are given an audio clip. Transcribe the spoken words as best you can."}]}]}
EOF
RESP="$(curl -fs --max-time 120 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @"$REQ")"
rm -f "$REQ"
CONTENT="$(echo "$RESP" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)"
echo "    model said: $(echo "$CONTENT" | head -c 220)"

# Audio actually drove the response: at least one distinctive spoken word
# should surface, and the encode must have produced audio soft tokens.
if echo "$CONTENT" | grep -qiE "fox|quick|brown|lazy|dog"; then
    ok "recognizes a spoken word from the audio"
else
    bad "recognizes a spoken word from the audio" "no expected word in: $(echo "$CONTENT" | head -c 200)"
fi
assert_contains "audio encode produced soft tokens" "audio soft tokens" "$(cat "$LOG")"
assert_contains "audio routed through the unified embedder" "0 vision +" "$(cat "$LOG")"

echo ""
echo -e "Gemma 4 12B unified audio: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
[ "$FAIL" -eq 0 ]
