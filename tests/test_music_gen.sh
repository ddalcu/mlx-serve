#!/usr/bin/env bash
# ACE-Step text2music on the ONE main server: headless boot -> load the music
# model by absolute path -> /v1/models shows the "music" capability -> POST
# /v1/audio/music-generations with a style prompt -> assert a valid 48 kHz
# stereo PCM16 WAV of the requested duration -> 400s (missing prompt, bad
# duration, TTS endpoint mismatch) -> SSE streaming -> coexist with a chat
# model -> unload. Proves the second audio backend routes end to end.
#
# Skips gracefully when no converted model is present. Convert with:
#   python3 tests/convert_acestep_weights.py --src-xl <dir> --src-main <dir>
#
# Usage: ACESTEP_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_music_gen.sh [port]
set -uo pipefail
PORT="${1:-11433}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MUSIC="${ACESTEP_MODEL:-$(ls -d ~/.mlx-serve/models/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit ~/.mlx-serve/models/local/acestep-v15-xl-turbo-8bit 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$MUSIC" ] || { echo "SKIP: no ACE-Step model (set ACESTEP_MODEL to a converted dir)"; exit 0; }
[ -f "$MUSIC/config.json" ] || { echo "SKIP: $MUSIC has no config.json (run tests/convert_acestep_weights.py)"; exit 0; }

# Headless: --model-dir anywhere; the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_music_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_music_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 3600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
MUSIC_ID="$(basename "$MUSIC")"

# 1. Load by absolute path -> ready with "audio" + "music" capabilities.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$MUSIC\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$MUSIC_ID' and x['state']=='ready']
assert m, 'ACE-Step not ready: '+json.dumps(d)
caps=m[0].get('capabilities',[])
assert 'audio' in caps and 'music' in caps, f'want audio+music caps, got {caps}'
print('PASS: load-model by path -> music model ready, capabilities', caps)
" || { echo "FAIL: ready music model missing audio/music capability"; exit 1; }

# 2. Generate (shortest valid duration -> smoke, not quality) -> valid WAV.
# Non-stream response mirrors /v1/audio/speech: raw audio/wav bytes.
cat > /tmp/test_music_req.json <<EOF
{"model":"$MUSIC_ID","prompt":"upbeat synthwave with driving bass and dreamy pads","duration_seconds":10,"seed":7}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_music_req.json -o /tmp/test_music_out.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: music gen http $code"; head -c 300 /tmp/test_music_out.wav; exit 1; }
python3 - /tmp/test_music_out.wav <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"not a WAV: {b[:12]!r}"
fmt, channels, rate = struct.unpack("<HHI", b[20:28])
bits = struct.unpack("<H", b[34:36])[0]
assert fmt == 1 and bits == 16, (fmt, bits)
assert channels == 2, f"want stereo, got {channels}"
assert rate == 48000, f"want 48 kHz, got {rate}"
n_samples = (len(b) - 44) // (2 * channels)
dur = n_samples / rate
assert abs(dur - 10.0) < 0.1, f"want ~10 s, got {dur:.2f} s"
# not digital silence: some sample must be nonzero
assert any(b[44:44+96000]), "output is all-zero audio"
print(f"PASS: /v1/audio/music-generations -> {len(b)} byte WAV, {dur:.2f} s 48 kHz stereo")
PY

# 3. Server survives the gen; the 400 family.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after music gen"; exit 1; }
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"duration_seconds\":10}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: missing prompt returned $code (want 400)"; exit 1; }
echo "PASS: missing 'prompt' -> 400"
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"duration_seconds\":5}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: duration 5 returned $code (want 400)"; exit 1; }
echo "PASS: out-of-range duration -> 400"
# The TTS endpoint against a music model is an explicit 400 (never a silent
# misinterpretation) — the wrong-backend guard on the shared audio slot.
code=$(api /v1/audio/speech -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"input\":\"hello\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: /v1/audio/speech on music model returned $code (want 400)"; exit 1; }
echo "PASS: /v1/audio/speech on a music model -> 400"

# 4. Streaming: SSE progress (encode/diffuse/decode stages) + base64 complete.
cat > /tmp/test_music_stream_req.json <<EOF
{"model":"$MUSIC_ID","prompt":"gentle acoustic folk guitar","duration_seconds":10,"seed":7,"stream":true}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_music_stream_req.json -o /tmp/test_music_stream.txt -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream music gen http $code"; exit 1; }
grep -q '"stage":"diffuse"' /tmp/test_music_stream.txt || { echo "FAIL: no diffuse progress in stream"; exit 1; }
grep -q '"type":"complete"' /tmp/test_music_stream.txt || { echo "FAIL: no complete event in stream"; exit 1; }
echo "PASS: streaming -> SSE progress + complete event"

# 5. Coexistence with a chat model.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while music model resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with music model also resident"
fi

# 6. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$MUSIC_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$MUSIC_ID']
assert m and m[0]['state']=='unloaded', 'music model should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> music model unloaded (stub retained)')
"

echo "ALL PASS: ACE-Step music gen (headless boot, load->gen->unload, WAV validity, 400s, streaming, coexistence)"
