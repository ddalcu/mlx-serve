#!/usr/bin/env bash
# Kokoro TTS end-to-end over HTTP.
#
#   KOKORO_MODEL=<dir> ./tests/test_kokoro_tts.sh [port]
#
# SKIPs without a converted checkpoint (tests/convert_kokoro_weights.py).
# Hermetic counterparts: the ~30 unit tests in src/kokoro.zig + src/kokoro_g2p.zig,
# and the KOKORO_FIXTURES parity oracles (durations exact, f0/n/asr cos 1.000000,
# audio cos 0.9968 vs a reference self-similarity floor of 0.9941-0.9960).
set -uo pipefail

PORT="${1:-11439}"
MODEL="${KOKORO_MODEL:-$HOME/.mlx-serve/models/hexgrad/Kokoro-82M-mlx-serve}"
BIN="./zig-out/bin/mlx-serve"
LOG="/tmp/kokoro-test-$PORT.log"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT

if [ ! -d "$MODEL" ]; then echo "SKIP: no Kokoro checkpoint at $MODEL"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; fi

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

speech() { curl -s -X POST "localhost:$PORT/v1/audio/speech" -H 'content-type: application/json' -d "$1"; }

# NOTE: build the JSON into a variable and pass it as ONE argument. Writing
# `check "label" "$(status "{\"a\":1}")" "400"` looks fine but bash mis-parses a
# double-quoted command substitution containing escaped quotes: the arg vector
# came out as 5 words instead of 3, so the expected code landed in $5 and every
# assertion compared against garbage (reported as `want '503'` with no 503
# anywhere in this file). Same class as any harness bug that silently shifts
# arguments — the tests "fail" while the server is perfectly correct.
expect_code() { # label, expected, body, [path]
  local path="${4:-/v1/audio/speech}" got
  got="$(curl -s -o /dev/null -w '%{http_code}' -X POST "localhost:$PORT$path" -H 'content-type: application/json' -d "$3")"
  check "$1" "$got" "$2"
}
secs()   { python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}')" "$1" 2>/dev/null || echo "0"; }

echo "== booting headless on :$PORT =="
# Discovery root comes from the MODEL, not a hardcoded home path — the
# library moved to an external drive and this discovered nothing.
ROOT="$(dirname "$(dirname "$MODEL")")"
[ -d "$ROOT" ] || ROOT="$HOME/.mlx-serve/models"
"$BIN" --serve --port "$PORT" --model-dir "$ROOT" --log-level info > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 40); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 0.5; done

ID="$(curl -s "localhost:$PORT/v1/models" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(next((m['id'] for m in d['data'] if 'kokoro' in m['id'].lower()), ''))")"
[ -n "$ID" ] && ok "[1] discovered as a media model: $ID" || { bad "[1] not discovered"; exit 1; }

echo "== [2] load =="
curl -s -X POST "localhost:$PORT/v1/load-model" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\"}" | grep -q '"state":"ready"' && ok "[2] loads" || bad "[2] load failed"

echo "== [3] synthesis =="
speech "{\"model\":\"$ID\",\"input\":\"The quick brown fox jumps over the lazy dog.\"}" > "$TMP/a.wav"
head -c 4 "$TMP/a.wav" | grep -q RIFF && ok "[3] returns a RIFF WAV" || bad "[3] not a WAV"
D="$(secs "$TMP/a.wav")"
python3 -c "import sys;sys.exit(0 if float('$D')>1.0 else 1)" && ok "[3] ${D}s of audio" || bad "[3] too short: ${D}s"
python3 - "$TMP/a.wav" <<'PY' && ok "[3] audio is non-silent and bounded" || bad "[3] silent or clipped"
import wave,sys,struct
w=wave.open(sys.argv[1]); n=w.getnframes()
d=struct.unpack(f"<{n}h", w.readframes(n))
peak=max(abs(x) for x in d)/32768
rms=(sum(x*x for x in d)/n)**0.5/32768
sys.exit(0 if 0.02 < peak < 1.0 and rms > 0.005 else 1)
PY

echo "== [4] voices are distinct and blendable =="
for v in af_heart am_michael bf_emma; do
  speech "{\"model\":\"$ID\",\"input\":\"Testing one two three.\",\"voice\":\"$v\"}" > "$TMP/$v.wav"
done
if cmp -s "$TMP/af_heart.wav" "$TMP/am_michael.wav"; then
  bad "[4] different voices produced identical audio"
else ok "[4] voices differ"; fi
speech "{\"model\":\"$ID\",\"input\":\"Testing one two three.\",\"voice\":\"af_heart,am_michael\"}" > "$TMP/blend.wav"
head -c 4 "$TMP/blend.wav" | grep -q RIFF && ok "[4] comma-separated blend synthesizes" || bad "[4] blend failed"
if cmp -s "$TMP/blend.wav" "$TMP/af_heart.wav"; then
  bad "[4] blend is a silent alias of its first voice"
else ok "[4] blend is a distinct voice"; fi

echo "== [5] speed scales duration =="
speech "{\"model\":\"$ID\",\"input\":\"The quick brown fox jumps over the lazy dog.\",\"speed\":0.8}" > "$TMP/slow.wav"
speech "{\"model\":\"$ID\",\"input\":\"The quick brown fox jumps over the lazy dog.\",\"speed\":1.5}" > "$TMP/fast.wav"
SLOW="$(secs "$TMP/slow.wav")"; FAST="$(secs "$TMP/fast.wav")"
python3 -c "import sys;sys.exit(0 if float('$SLOW')>float('$FAST')*1.4 else 1)" \
  && ok "[5] speed 0.8=${SLOW}s > 1.5=${FAST}s" || bad "[5] speed had no effect (${SLOW}s vs ${FAST}s)"

echo "== [6] unsupported controls are NAMED 400s, never silently ignored =="
BODY_REF="{\"model\":\"$ID\",\"input\":\"hi\",\"ref_audio\":\"AAAA\"}"
BODY_BADVOICE="{\"model\":\"$ID\",\"input\":\"hi\",\"voice\":\"nope\"}"
BODY_BADBLEND="{\"model\":\"$ID\",\"input\":\"hi\",\"voice\":\"af_heart,nope\"}"
BODY_BADSPEED="{\"model\":\"$ID\",\"input\":\"hi\",\"speed\":99}"
BODY_EMPTY="{\"model\":\"$ID\",\"input\":\"\"}"
BODY_MUSIC="{\"model\":\"$ID\",\"prompt\":\"jazz\"}"
BODY_CHAT="{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"

expect_code "[6] ref_audio (no cloning)" 400 "$BODY_REF"
expect_code "[6] unknown voice"          400 "$BODY_BADVOICE"
expect_code "[6] blend w/ one bad voice" 400 "$BODY_BADBLEND"
expect_code "[6] speed out of range"     400 "$BODY_BADSPEED"
expect_code "[6] empty input"            400 "$BODY_EMPTY"
expect_code "[6] music endpoint on TTS"  400 "$BODY_MUSIC" /v1/audio/music-generations
expect_code "[6] chat on a TTS model"    400 "$BODY_CHAT"  /v1/chat/completions

echo "== [7] streaming =="
speech "{\"model\":\"$ID\",\"input\":\"streaming test\",\"stream\":true}" | grep -q '"type":"complete"' \
  && ok "[7] SSE complete event" || bad "[7] no complete event"

echo "== [8] text normalization reaches the audio (numbers must not vanish) =="
# Kokoro's vocab has NO digits: an unexpanded number is silently dropped by the
# encoder, so "42" would just disappear from the speech rather than error.
speech "{\"model\":\"$ID\",\"input\":\"It costs 42 dollars.\"}" > "$TMP/num.wav"
speech "{\"model\":\"$ID\",\"input\":\"It costs dollars.\"}" > "$TMP/nonum.wav"
NUM="$(secs "$TMP/num.wav")"; NONUM="$(secs "$TMP/nonum.wav")"
python3 -c "import sys;sys.exit(0 if float('$NUM')>float('$NONUM')+0.15 else 1)" \
  && ok "[8] '42' adds speech (${NUM}s vs ${NONUM}s)" || bad "[8] number vanished (${NUM}s vs ${NONUM}s)"

echo "== [9] unload =="
curl -s -X POST "localhost:$PORT/v1/unload-model" -H 'content-type: application/json' \
  -d "{\"model\":\"$ID\"}" >/dev/null && ok "[9] unloads" || bad "[9] unload failed"

echo
echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
