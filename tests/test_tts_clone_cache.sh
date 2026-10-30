#!/usr/bin/env bash
# test_tts_clone_cache.sh — the speaker-embedding cache (voice-clone path) never changes a voice. [1] same clip + text
# twice -> byte-identical WAVs and a `speaker embedding: cache hit` line (equality alone cannot see a silent uncached
# fallback; the first request is the uncached miss); [2] a different clip -> a different WAV (no key collision);
# [3] `warm_only` pre-warm: miss then hit, the first synthesis after a warm is a hit with a byte-identical WAV, and warm_only without
# ref_audio is a named 400. Synthetic deterministic clips.
#
#   TTS_MODEL=<Qwen3-TTS -Base dir> ./tests/test_tts_clone_cache.sh [port]   # SKIPs without one
#
# Hermetic twins: the `spk cache` tests in src/plugins/qwen3_tts/tts.zig (+ the TTS_TEST_MODEL-gated identity test).

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

cmp -s "$TMP/on_a1.wav" "$TMP/on_a2.wav" || { echo "FAIL: [1/3] same clip twice -> WAVs differ"; exit 1; }
grep -q "speaker embedding: cache hit" "$LOG_ON" || { echo "FAIL: [1/3] no cache-hit engagement line"; exit 1; }
echo "PASS: [1/3] same clip -> byte-identical WAV + cache hit engaged"
cmp -s "$TMP/on_a1.wav" "$TMP/on_b.wav" && { echo "FAIL: [2/3] different clip -> identical WAV"; exit 1; }
echo "PASS: [2/3] different clip -> different WAV"

# ── Boot 2: warm_only pre-warm ──
LOG_WARM="$TMP/server_warm.log"
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_WARM" 2>&1 &
SRV=$!
wait_up "$LOG_WARM"
W1=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' --data @"$TMP/warm_b.json")
echo "$W1" | grep -q '"cache":"miss"' || { echo "FAIL: [3/3] first warm did not report miss: $W1"; exit 1; }
W2=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' --data @"$TMP/warm_b.json")
echo "$W2" | grep -q '"cache":"hit"' || { echo "FAIL: [3/3] second warm did not report hit: $W2"; exit 1; }
speak "$TMP/req_b.json" "$TMP/warm_b1.wav"
grep -q "speaker embedding: cache hit" "$LOG_WARM" || { echo "FAIL: [3/3] first synthesis after warm was not a cache hit"; exit 1; }
cmp -s "$TMP/warm_b1.wav" "$TMP/on_b.wav" || { echo "FAIL: [3/3] warmed WAV differs from unwarmed boot's"; exit 1; }
CODE=$(curl -s -o "$TMP/warm_err.json" -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' --data @"$TMP/warm_noref.json")
[ "$CODE" = "400" ] || { echo "FAIL: [3/3] warm_only without ref_audio -> $CODE (want 400)"; exit 1; }
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""
echo "PASS: [3/3] warm_only: miss then hit, first synthesis hits, named 400 without ref"
