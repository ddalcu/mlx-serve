#!/usr/bin/env bash
# 8-bit quantized Qwen3-TTS end-to-end: boot the unified server with an
# mlx-community *-8bit checkpoint (affine 8-bit talker + code predictor,
# dense codec/speaker-encoder), then:
#   1) plain /v1/audio/speech → plausible WAV
#   2) /v1/audio/speech with ref_audio (zero-shot voice clone) → plausible WAV
# SKIPs cleanly when the 8-bit model isn't downloaded.
# Usage: [TTS_QUANT_MODEL=<dir>] ./tests/test_tts_quant.sh [port]
set -euo pipefail
PORT="${1:-11319}"
MODEL="${TTS_QUANT_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit}"
[ -d "$MODEL" ] || { echo "SKIP: no 8-bit qwen3_tts model at $MODEL (set TTS_QUANT_MODEL)"; exit 0; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_tts_quant_server.log

"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null || true" EXIT
for i in $(seq 1 120); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server exited during boot"; cat "$LOG"; exit 1; }
  sleep 1
done
curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server did not start"; cat "$LOG"; exit 1; }

# 1) Plain synthesis through the quantized talker + code predictor.
code=$(curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"tts","input":"This is the eight bit native server test."}' -o /tmp/test_tts_quant.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; tail -40 "$LOG"; exit 1; }
[ "$(head -c 4 /tmp/test_tts_quant.wav)" = "RIFF" ] || { echo "FAIL: not a WAV"; exit 1; }
sz=$(wc -c < /tmp/test_tts_quant.wav); [ "$sz" -gt 40000 ] || { echo "FAIL: WAV too small ($sz)"; exit 1; }
echo "PASS: 8-bit /v1/audio/speech -> $sz byte WAV"

# 2) Zero-shot cloning: a synthetic 24 kHz mono reference clip. The speaker
# encoder is dense in the 8-bit repos; this exercises the speaker-embedding
# splice through the QUANTIZED talker prefix.
python3 - <<'PY'
import base64, json, math, struct
sr = 24000
n = sr  # 1 s
samples = []
for i in range(n):
    t = i / sr
    env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * t)
    x = env * (0.3 * math.sin(2 * math.pi * 160 * t) + 0.15 * math.sin(2 * math.pi * 320 * t))
    samples.append(int(max(-1.0, min(1.0, x)) * 32767))
pcm = struct.pack("<%dh" % n, *samples)
hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16, b"data", len(pcm))
body = {"model": "tts", "input": "Cloning check for the quantized checkpoint.", "ref_audio": base64.b64encode(hdr + pcm).decode()}
open("/tmp/test_tts_quant_req.json", "w").write(json.dumps(body))
PY
code=$(curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
  -d @/tmp/test_tts_quant_req.json -o /tmp/test_tts_quant_clone.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: clone http $code"; tail -40 "$LOG"; exit 1; }
[ "$(head -c 4 /tmp/test_tts_quant_clone.wav)" = "RIFF" ] || { echo "FAIL: clone not a WAV"; exit 1; }
sz=$(wc -c < /tmp/test_tts_quant_clone.wav); [ "$sz" -gt 40000 ] || { echo "FAIL: clone WAV too small ($sz)"; exit 1; }
grep -q "ignoring ref_audio" "$LOG" && { echo "FAIL: speaker encoder missing — clone silently downgraded"; exit 1; }
echo "PASS: 8-bit voice clone -> $sz byte WAV"
echo "ALL PASS"
