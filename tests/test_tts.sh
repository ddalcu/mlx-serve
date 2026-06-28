#!/usr/bin/env bash
# Native Qwen3-TTS /v1/audio/speech endpoint smoke test.
# Usage: TTS_MODEL=<dir> ./tests/test_tts.sh [port]
set -euo pipefail
PORT="${1:-11377}"
MODEL="${TTS_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16/snapshots/* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no qwen3_tts model (set TTS_MODEL)"; exit 0; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
"$BIN" --model "$MODEL" --serve --port "$PORT" >/tmp/test_tts_server.log 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null || true" EXIT
for i in $(seq 1 90); do grep -q "TTS server listening" /tmp/test_tts_server.log && break; sleep 1; done
grep -q "TTS server listening" /tmp/test_tts_server.log || { echo "FAIL: server did not start"; cat /tmp/test_tts_server.log; exit 1; }
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"tts","input":"This is a native server test."}' -o /tmp/test_tts.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; exit 1; }
[ "$(head -c 4 /tmp/test_tts.wav)" = "RIFF" ] || { echo "FAIL: not a WAV"; exit 1; }
sz=$(wc -c < /tmp/test_tts.wav); [ "$sz" -gt 40000 ] || { echo "FAIL: WAV too small ($sz)"; exit 1; }
echo "PASS: /v1/audio/speech -> $sz byte WAV"
