#!/usr/bin/env bash
# Speaker-embedding cache (voice-clone path) guard: the cache must never change
# what a voice sounds like.
#   [1/4] same clip + text twice -> byte-identical WAVs, second request logs a
#         cache hit (engagement — output equality alone can't see a silent
#         fallback to the uncached path)
#   [2/4] a different clip -> a different WAV (no key-collision wrong-voice hit)
#   [3/4] MLX_SERVE_TTS_SPK_CACHE=0 boot -> byte-identical to the cache-on
#         output (the kill switch restores the uncached path exactly) and no
#         cache-hit line ever appears
#   [4/4] cache-on and cache-off never disagree on the different-clip output
#   [5/5] warm_only pre-warm (docs/qwentts-cache.md): warm b -> {"cache":"miss"},
#         warm b again -> {"cache":"hit"}, the FIRST synthesis after the warm
#         logs a cache hit (the first-sentence win the endpoint exists for),
#         its WAV is byte-identical to the unwarmed boot's, and warm_only
#         without ref_audio is a named 400
# Usage: TTS_MODEL=<dir> ./tests/test_tts_clone_cache.sh [port]
set -euo pipefail
PORT="${1:-11378}"
MODEL="${TTS_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3-TTS-12Hz-*-Base-* 2>/dev/null | head -1 || true)}"
[ -n "$MODEL" ] || { echo "SKIP: no qwen3_tts model (set TTS_MODEL)"; exit 0; }
[ -d "$MODEL" ] || { echo "FAIL: TTS_MODEL points at a missing dir: $MODEL" >&2; exit 1; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
TMP=$(mktemp -d /tmp/tts_clone_cache.XXXXXX)
SRV=""
trap 'kill $SRV 2>/dev/null || true; rm -rf "$TMP"' EXIT

# Two deterministic synthetic 24 kHz mono clips with different spectra (two
# distinct "voices") + the request bodies.
python3 - "$TMP" <<'PY'
import sys, wave, math, struct, json, base64, os
tmp = sys.argv[1]
def clip(path, f0):
    frames = bytearray()
    for i in range(24000):
        t = i / 24000.0
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * t)
        x = env * (0.3 * math.sin(2 * math.pi * f0 * t) + 0.15 * math.sin(2 * math.pi * 2 * f0 * t))
        frames += struct.pack('<h', int(x * 32767))
    w = wave.open(path, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(bytes(frames)); w.close()
clip(os.path.join(tmp, 'a.wav'), 160.0)
clip(os.path.join(tmp, 'b.wav'), 240.0)
for name in ('a', 'b'):
    b64 = base64.b64encode(open(os.path.join(tmp, name + '.wav'), 'rb').read()).decode()
    body = {'input': 'The cache must never change what a voice sounds like.', 'ref_audio': b64}
    json.dump(body, open(os.path.join(tmp, 'req_' + name + '.json'), 'w'))
    json.dump({'warm_only': True, 'ref_audio': b64}, open(os.path.join(tmp, 'warm_' + name + '.json'), 'w'))
json.dump({'warm_only': True}, open(os.path.join(tmp, 'warm_noref.json'), 'w'))
PY

wait_up() { # $1 = log
  for i in $(seq 1 90); do grep -q "Server listening" "$1" && return 0; sleep 1; done
  echo "FAIL: server did not start"; tail -30 "$1"; exit 1
}
speak() { # $1 = request json, $2 = out wav
  local code
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
    --data @"$1" -o "$2" -w "%{http_code}")
  [ "$code" = "200" ] || { echo "FAIL: http $code for $1"; exit 1; }
  [ "$(head -c 4 "$2")" = "RIFF" ] || { echo "FAIL: not a WAV ($1)"; exit 1; }
}

# ── Boot 1: cache on (default) ──
LOG_ON="$TMP/server_on.log"
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_ON" 2>&1 &
SRV=$!
wait_up "$LOG_ON"
speak "$TMP/req_a.json" "$TMP/on_a1.wav"
speak "$TMP/req_a.json" "$TMP/on_a2.wav"
speak "$TMP/req_b.json" "$TMP/on_b.wav"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""

cmp -s "$TMP/on_a1.wav" "$TMP/on_a2.wav" || { echo "FAIL: [1/4] same clip twice -> WAVs differ"; exit 1; }
grep -q "speaker embedding: cache hit" "$LOG_ON" || { echo "FAIL: [1/4] no cache-hit engagement line"; exit 1; }
echo "PASS: [1/4] same clip -> byte-identical WAV + cache hit engaged"
cmp -s "$TMP/on_a1.wav" "$TMP/on_b.wav" && { echo "FAIL: [2/4] different clip -> identical WAV"; exit 1; }
echo "PASS: [2/4] different clip -> different WAV"

# ── Boot 2: kill switch ──
LOG_OFF="$TMP/server_off.log"
MLX_SERVE_TTS_SPK_CACHE=0 "$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_OFF" 2>&1 &
SRV=$!
wait_up "$LOG_OFF"
speak "$TMP/req_a.json" "$TMP/off_a1.wav"
speak "$TMP/req_a.json" "$TMP/off_a2.wav"
speak "$TMP/req_b.json" "$TMP/off_b.wav"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""

grep -q "speaker embedding: cache hit" "$LOG_OFF" && { echo "FAIL: [3/4] kill switch set but cache still engaged"; exit 1; }
cmp -s "$TMP/on_a1.wav" "$TMP/off_a1.wav" || { echo "FAIL: [3/4] cache-on WAV differs from cache-off WAV"; exit 1; }
cmp -s "$TMP/off_a1.wav" "$TMP/off_a2.wav" || { echo "FAIL: [3/4] uncached path not deterministic"; exit 1; }
echo "PASS: [3/4] kill switch: uncached path engaged, output byte-identical to cached"
cmp -s "$TMP/on_b.wav" "$TMP/off_b.wav" || { echo "FAIL: [4/4] different-clip WAV disagrees across cache modes"; exit 1; }
echo "PASS: [4/4] cache modes agree on the second voice too"

# ── Boot 3: warm_only pre-warm ──
LOG_WARM="$TMP/server_warm.log"
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_WARM" 2>&1 &
SRV=$!
wait_up "$LOG_WARM"
W1=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' --data @"$TMP/warm_b.json")
echo "$W1" | grep -q '"cache":"miss"' || { echo "FAIL: [5/5] first warm did not report miss: $W1"; exit 1; }
W2=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' --data @"$TMP/warm_b.json")
echo "$W2" | grep -q '"cache":"hit"' || { echo "FAIL: [5/5] second warm did not report hit: $W2"; exit 1; }
speak "$TMP/req_b.json" "$TMP/warm_b1.wav"
grep -q "speaker embedding: cache hit" "$LOG_WARM" || { echo "FAIL: [5/5] first synthesis after warm was not a cache hit"; exit 1; }
cmp -s "$TMP/warm_b1.wav" "$TMP/on_b.wav" || { echo "FAIL: [5/5] warmed WAV differs from unwarmed boot's"; exit 1; }
CODE=$(curl -s -o "$TMP/warm_err.json" -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' --data @"$TMP/warm_noref.json")
[ "$CODE" = "400" ] || { echo "FAIL: [5/5] warm_only without ref_audio -> $CODE (want 400)"; exit 1; }
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""
echo "PASS: [5/5] warm_only: miss then hit, first synthesis hits, named 400 without ref"
