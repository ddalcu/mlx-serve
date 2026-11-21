#!/usr/bin/env bash
# Breeze-TTS-2 end-to-end over HTTP.
#
#   ./tests/test_breeze_tts.sh [port]
#
# SKIPs without a pack. The pack must carry `model_type: breeze_tts` (the
# mlx-community spelling). Hermetic counterparts: the unit tests in
# src/plugins/breeze_tts/engine.zig (voice slots, frame budget, sampler
# determinism) and the codec oracle test in mimi.zig.
set -uo pipefail

PORT="${1:-11441}"
source "$(dirname "$0")/_lib_models.sh"
MODEL="${BREEZE_MODEL:-$(find_model mlx-community/Breeze-TTS-2-mlx || true)}"
BIN="./zig-out/bin/mlx-serve"
LOG="/tmp/breeze-test-$PORT.log"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT

if [ ! -d "$MODEL" ]; then echo "SKIP: no Breeze pack at $MODEL"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; fi

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

"$BIN" --model "$MODEL" --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
# The port answers before the engine is ready (503 while loading): wait for a real 200.
for _ in $(seq 1 180); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/health")" = "200" ] && break
  kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server died during load (see $LOG)"; exit 1; }
  sleep 1
done

speech() { curl -s -X POST "localhost:$PORT/v1/audio/speech" -H 'content-type: application/json' -d "$1" -o "$2" -w "%{http_code}"; }
code()   { curl -s -X POST "localhost:$PORT/v1/audio/speech" -H 'content-type: application/json' -d "$1" -o /dev/null -w "%{http_code}"; }
msg()    { curl -s -X POST "localhost:$PORT/v1/audio/speech" -H 'content-type: application/json' -d "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("error",{}).get("message",""))'; }

check "model loads + /health" "$(curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/health)" "200"

PLAIN_TEXT="Hello world, this is a Breeze test."
C=$(speech "{\"input\":\"$PLAIN_TEXT\",\"voice\":\"S0\"}" "$TMP/plain.wav")
check "plain S0 200" "$C" "200"
SZ=$(wc -c < "$TMP/plain.wav" | tr -d ' ')
if [ "$SZ" -gt 20000 ]; then ok "plain S0 wav audio ($SZ bytes)"; else bad "plain S0 wav too small ($SZ bytes)"; fi
if python3 - "$TMP/plain.wav" <<'EOF'
import sys, wave, struct, math
w = wave.open(sys.argv[1])
assert w.getnchannels() == 1 and w.getframerate() == 24000
d = w.readframes(w.getnframes())
n = len(d) // 2
s = struct.unpack(f"<{n}h", d)
rms = math.sqrt(sum(x * x for x in s) / n) / 32768
assert 0.01 < rms < 0.99, f"rms {rms} not speech-like"
EOF
then ok "plain S0 24 kHz mono, speech-like RMS"; else bad "plain S0 wav header/RMS check"; fi

C=$(speech '{"input":"Weather report: clear skies.","voice":"S2","instruction":"calm"}' "$TMP/ins.wav")
check "instruction turn 200" "$C" "200"
SZ=$(wc -c < "$TMP/ins.wav" | tr -d ' ')
if [ "$SZ" -gt 20000 ]; then ok "instruction wav audio ($SZ bytes)"; else bad "instruction wav too small ($SZ bytes)"; fi

C=$(speech '{"input":"Deterministic seed.","voice":"S1","seed":7}' "$TMP/s1.wav")
check "seeded 200" "$C" "200"
C=$(speech '{"input":"Deterministic seed.","voice":"S1","seed":7}' "$TMP/s2.wav")
check "seeded repeat 200" "$C" "200"
if cmp -s "$TMP/s1.wav" "$TMP/s2.wav"; then ok "same seed same bytes"; else bad "seed determinism"; fi

# Voice cloning: the plain turn is the reference, and we know its exact transcript.
REF=$(base64 -i "$TMP/plain.wav")
C=$(speech "{\"input\":\"Cloning through the codec encoder.\",\"ref_audio\":\"$REF\",\"ref_text\":\"$PLAIN_TEXT\",\"seed\":3}" "$TMP/clone.wav")
check "clone 200" "$C" "200"
SZ=$(wc -c < "$TMP/clone.wav" | tr -d ' ')
if [ "$SZ" -gt 20000 ]; then ok "clone wav audio ($SZ bytes)"; else bad "clone wav too small ($SZ bytes)"; fi

# Streamed answers carry frame progress before the complete event.
SC=$(curl -sN -m 180 -X POST "localhost:$PORT/v1/audio/speech" -H 'content-type: application/json' \
     -d '{"input":"Progress over SSE.","stream":true}' -o "$TMP/sse.txt" -w "%{http_code}")
if [ "$SC" = "200" ] && grep -q '"stage":"breeze"' "$TMP/sse.txt" && grep -q '"type":"complete"' "$TMP/sse.txt";
then ok "stream emits breeze progress"; else bad "stream progress events (http $SC, $(wc -c < "$TMP/sse.txt" | tr -d ' ') bytes)"; fi

check "unknown voice 400" "$(code '{"input":"x","voice":"S99"}')" "400"
check "missing input 400" "$(code '{"voice":"S0"}')" "400"
check "empty input 400" "$(code '{"input":"","voice":"S0"}')" "400"
check "clone without ref_text 400" "$(code "{\"input\":\"x\",\"ref_audio\":\"$REF\"}")" "400"
if msg '{"input":"x","language":"en"}' | grep -q "'language'"; then ok "language refused by name"; else bad "language refusal names the field"; fi
if msg '{"input":"x","voice":"S99"}' | grep -q "'voice'"; then ok "bad voice refused by name"; else bad "voice refusal names the field"; fi
check "bad base64 400" "$(code '{"input":"x","ref_audio":"!!!","ref_text":"y"}')" "400"

echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = "0" ]
