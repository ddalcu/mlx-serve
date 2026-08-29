#!/usr/bin/env bash
# ACE-Step text2music on the ONE main server: headless boot -> load the music
# model by absolute path -> /v1/models shows the "music" capability -> POST
# /v1/audio/music-generations with a style prompt -> assert a valid 48 kHz
# stereo PCM16 WAV of the requested duration -> ref_audio -> tasks complete +
# cover (source length kept, engagement lines) -> 400s (missing prompt, bad
# duration, task/src_audio pairing, TTS endpoint mismatch) -> SSE streaming ->
# coexist with a chat model -> unload. Proves the second audio backend routes
# end to end.
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

# ── Reference audio (#259): the previous arm's WAV fed back as `ref_audio`.
# The guard is the ENGAGEMENT line, never the output — a silently ignored
# clip still yields a perfectly good track.
base64 < /tmp/test_music_out.wav | tr -d '\n' > /tmp/test_music_out.b64
# mkreq <out.json> '<json object of extra fields>' — base64 clips are read from
# FILES (a 10 s WAV's base64 is past ARG_MAX): the value "@b64" expands to it.
mkreq() { python3 - "$1" "$2" "$MUSIC_ID" <<'PY'
import json, sys
extra = json.loads(sys.argv[2])
b64 = open('/tmp/test_music_out.b64').read()
body = {'model': sys.argv[3]}
body.update({k: (b64 if v == '@b64' else v) for k, v in extra.items()})
json.dump(body, open(sys.argv[1], 'w'))
PY
}
mkreq /tmp/test_music_ref_req.json '{"prompt":"lo-fi hip hop, mellow","duration_seconds":10,"seed":3,"ref_audio":"@b64"}'
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' --data @/tmp/test_music_ref_req.json -o /tmp/test_music_ref.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: ref_audio music gen http $code"; head -c 300 /tmp/test_music_ref.wav; exit 1; }
grep -q "\[acestep\] reference audio: 750 latent frames" /tmp/test_music_server.log \
  || { echo "FAIL: reference audio not engaged (no '[acestep] reference audio: 750 latent frames' log line)"; grep "reference" /tmp/test_music_server.log | head -3; exit 1; }
echo "PASS: ref_audio -> 200 + timbre slot fed from the clip (750 latent frames)"
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"duration_seconds\":10,\"ref_audio\":\"!!notbase64\"}" -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] && grep -q "ref_audio" /tmp/test_music_err.txt || { echo "FAIL: bad ref_audio returned $code (want named 400)"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: bad ref_audio base64 -> named 400"
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"duration_seconds\":10,\"ref_audio\":\"AAAA\"}" -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] && grep -q "WAV" /tmp/test_music_err.txt || { echo "FAIL: non-WAV ref_audio returned $code (want named 400)"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: non-WAV ref_audio -> named 400"
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"duration_seconds\":10}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: missing prompt returned $code (want 400)"; exit 1; }
echo "PASS: missing 'prompt' -> 400"
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"duration_seconds\":5}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: duration 5 returned $code (want 400)"; exit 1; }
echo "PASS: out-of-range duration -> 400"
# `instrumental` is the EXPLICIT spelling of what an empty lyric block already
# meant on ACE-Step, so it must resolve to the same [Instrumental] marker — and
# sending it beside real lyrics is a NAMED 400 on BOTH backends (one predicate).
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"instrumental\":true,\"lyrics\":\"la la\"}" \
  -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: instrumental+lyrics returned $code (want 400)"; cat /tmp/test_music_err.txt; exit 1; }
grep -q "instrumental" /tmp/test_music_err.txt \
  || { echo "FAIL: conflict 400 does not NAME 'instrumental'"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: instrumental beside lyrics -> named 400"
# The TTS endpoint against a music model is an explicit 400 (never a silent
# misinterpretation) — the wrong-backend guard on the shared audio slot.
code=$(api /v1/audio/speech -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"input\":\"hello\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: /v1/audio/speech on music model returned $code (want 400)"; exit 1; }
echo "PASS: /v1/audio/speech on a music model -> 400"

# ── Tasks: `complete` (vocal-to-BGM) needs no new weights; `cover` reads the
# FSQ tokenizer (fsq.safetensors). Source = the first arm's 10 s WAV. Guards
# are the ENGAGEMENT lines + the length contract (track == source length).
mkreq /tmp/test_music_complete_req.json '{"prompt":"full band arrangement","duration_seconds":60,"seed":5,"task":"complete","src_audio":"@b64","track_classes":["bass","drums"]}'
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' --data @/tmp/test_music_complete_req.json -o /tmp/test_music_complete.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: complete task http $code"; head -c 300 /tmp/test_music_complete.wav; exit 1; }
grep -q "\[acestep\] complete: 250 source frames as context, classes=BASS | DRUMS" /tmp/test_music_server.log \
  || { echo "FAIL: complete task not engaged (no '[acestep] complete: 250 source frames' line)"; grep "acestep\] complete" /tmp/test_music_server.log | head -3; exit 1; }
python3 - /tmp/test_music_complete.wav <<'PY'
import sys
b = open(sys.argv[1], "rb").read()
dur = ((len(b) - 44) // 4) / 48000
assert abs(dur - 10.0) < 0.1, f"complete must keep the SOURCE length (10 s), got {dur:.2f} s"
PY
echo "PASS: task complete -> 200, source latent as context, BASS | DRUMS classes, source length kept"
if [ -f "$MUSIC/fsq.safetensors" ]; then
  mkreq /tmp/test_music_cover_req.json '{"prompt":"orchestral cover, strings and brass","duration_seconds":60,"seed":5,"task":"cover","src_audio":"@b64","cover_strength":0.75,"cover_noise_strength":0.5}'
  code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' --data @/tmp/test_music_cover_req.json -o /tmp/test_music_cover.wav -w "%{http_code}")
  [ "$code" = "200" ] || { echo "FAIL: cover task http $code"; head -c 300 /tmp/test_music_cover.wav; exit 1; }
  grep -q "\[acestep\] cover: source 250 frames -> 50 codes" /tmp/test_music_server.log \
    || { echo "FAIL: cover not engaged (no '[acestep] cover: source 250 frames -> 50 codes' line)"; grep "acestep\] cover" /tmp/test_music_server.log | head -3; exit 1; }
  grep -q "\[acestep\] cover: switching to text2music conditioning" /tmp/test_music_server.log \
    || { echo "FAIL: cover_strength<1 did not switch condition sets"; exit 1; }
  grep -q "\[acestep\] cover: noise_strength=0.50 -> start at t=0.500" /tmp/test_music_server.log \
    || { echo "FAIL: cover_noise_strength start point not applied"; grep "noise_strength" /tmp/test_music_server.log | head -3; exit 1; }
  python3 - /tmp/test_music_cover.wav <<'PY'
import sys
b = open(sys.argv[1], "rb").read()
dur = ((len(b) - 44) // 4) / 48000
assert abs(dur - 10.0) < 0.1, f"cover must keep the SOURCE length (10 s), got {dur:.2f} s"
PY
  echo "PASS: task cover -> 200, 50 FSQ codes, strength switch + noise start, source length kept"
else
  echo "SKIP: cover arm ($MUSIC has no fsq.safetensors — tests/convert_acestep_weights.py --fsq-only)"
fi
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"task\":\"cover\"}" -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] && grep -q "src_audio" /tmp/test_music_err.txt || { echo "FAIL: cover without src_audio returned $code (want named 400)"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: cover without src_audio -> named 400"
mkreq /tmp/test_music_bad_req.json '{"prompt":"jazz","src_audio":"@b64"}'
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  --data @/tmp/test_music_bad_req.json -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] && grep -q "src_audio" /tmp/test_music_err.txt || { echo "FAIL: src_audio on text2music returned $code (want named 400)"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: src_audio on text2music -> named 400"
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"task\":\"remix\"}" -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] && grep -q "task" /tmp/test_music_err.txt || { echo "FAIL: bad task returned $code (want named 400)"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: unknown task -> named 400"
mkreq /tmp/test_music_bad_req.json '{"prompt":"jazz","task":"complete","src_audio":"@b64","track_classes":["kazoo"]}'
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  --data @/tmp/test_music_bad_req.json -o /tmp/test_music_err.txt -w "%{http_code}")
[ "$code" = "400" ] && grep -q "track_classes" /tmp/test_music_err.txt || { echo "FAIL: unknown track class returned $code (want named 400)"; cat /tmp/test_music_err.txt; exit 1; }
echo "PASS: unknown track_classes entry -> named 400"

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
