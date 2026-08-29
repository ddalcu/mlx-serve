#!/usr/bin/env bash
# MiniMax Music 3 text2music on the ONE main server: headless boot -> load the
# pack by absolute path -> /v1/models shows "audio"+"music" capabilities ->
# POST /v1/audio/music-generations with caption+lyrics -> assert a valid
# 44.1 kHz stereo PCM16 WAV -> the 400 family (missing prompt, missing lyrics,
# ACE-Step-only fields refused BY NAME, bad duration/steps, TTS endpoint
# mismatch) -> SSE streaming with per-frame progress -> chat coexistence ->
# unload. Proves the third audio backend routes end to end.
#
# Skips gracefully when no converted pack is present. Convert with:
#   python3 scripts/convert_music3_weights.py <src> <out>
#
# Usage: MUSIC3_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_music3_gen.sh [port]
set -uo pipefail
PORT="${1:-11437}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MUSIC="${MUSIC3_MODEL:-$(ls -d ~/.mlx-serve/models/ddalcu/MiniMax-Music3-MLX-Serve-8bit 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$MUSIC" ] || { echo "SKIP: no MiniMax Music 3 pack (set MUSIC3_MODEL to a converted dir)"; exit 0; }
[ -f "$MUSIC/config.json" ] || { echo "SKIP: $MUSIC has no config.json (run scripts/convert_music3_weights.py)"; exit 0; }
[ -f "$MUSIC/vocoder.safetensors" ] || { echo "SKIP: $MUSIC is incomplete (no vocoder.safetensors marker)"; exit 0; }

# Headless: --model-dir anywhere; the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_music3_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_music3_server.log; exit 1; }
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
assert m, 'Music3 not ready: '+json.dumps(d)
caps=m[0].get('capabilities',[])
assert 'audio' in caps and 'music' in caps, f'want audio+music caps, got {caps}'
print('PASS: load-model by path -> music model ready, capabilities', caps)
" || { echo "FAIL: ready music model missing audio/music capability"; exit 1; }

# 2. The 400 family — cheap, BEFORE the expensive generation.
b400() { # label body
  local code
  code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
    -d "$2" -o /tmp/test_music3_err.txt -w "%{http_code}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat /tmp/test_music3_err.txt; exit 1; }
  echo "PASS: $1 -> 400"
}
b400 "missing prompt" "{\"model\":\"$MUSIC_ID\",\"lyrics\":\"[verse]\\nla la\",\"duration_seconds\":5}"
b400 "missing lyrics" "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"duration_seconds\":5}"
b400 "duration 0" "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"lyrics\":\"la\",\"duration_seconds\":0}"
b400 "duration 999" "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"lyrics\":\"la\",\"duration_seconds\":999}"
b400 "steps 2" "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"lyrics\":\"la\",\"steps\":2}"
# `instrumental` and real lyrics are contradictory — a NAMED 400, never a
# silent winner.
b400 "instrumental beside lyrics" "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"instrumental\":true,\"lyrics\":\"[verse]\\nla la\"}"
grep -q "instrumental" /tmp/test_music3_err.txt \
  || { echo "FAIL: conflict 400 does not NAME 'instrumental'"; cat /tmp/test_music3_err.txt; exit 1; }
# The missing-lyrics 400 must point at the way out, or the flag is undiscoverable.
grep -q 'instrumental' <(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\"}") \
  || { echo "FAIL: missing-lyrics 400 does not mention the instrumental escape hatch"; exit 1; }
echo "PASS: missing-lyrics 400 names the instrumental escape hatch"
# ACE-Step-only fields have NO music3 equivalent — refused by name, never ignored.
# (bpm/keyscale are NOT in this list: since 259bd15 they ride the caption here.)
for f in '"timesignature":"4/4"' '"vocal_language":"en"' '"ref_audio":"AAAA"' '"src_audio":"AAAA"' '"task":"cover"'; do
  b400 "ACE-Step field ${f%%:*}" "{\"model\":\"$MUSIC_ID\",\"prompt\":\"jazz\",\"lyrics\":\"la\",$f}"
  grep -q 'ACE-Step' /tmp/test_music3_err.txt || grep -q 'no equivalent' /tmp/test_music3_err.txt \
    || { echo "FAIL: unsupported-field 400 does not NAME the problem"; cat /tmp/test_music3_err.txt; exit 1; }
done
# The TTS endpoint against a music model is an explicit 400.
code=$(api /v1/audio/speech -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"input\":\"hello\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: /v1/audio/speech on music model returned $code (want 400)"; exit 1; }
echo "PASS: /v1/audio/speech on a music model -> 400"

# 3. Generate (short duration + few steps -> smoke, not quality) -> valid WAV.
# duration is an UPPER bound: the LLM may emit <|audio_end|> earlier, so the
# assertion is a ceiling + non-triviality, not an exact length.
cat > /tmp/test_music3_req.json <<EOF
{"model":"$MUSIC_ID","prompt":"upbeat synthwave with driving bass and dreamy pads","lyrics":"[verse]\nneon lights across the bay\n[chorus]\nwe run all night","duration_seconds":8,"steps":8,"seed":7}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_music3_req.json -o /tmp/test_music3_out.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: music gen http $code"; head -c 300 /tmp/test_music3_out.wav; tail -20 /tmp/test_music3_server.log; exit 1; }
python3 - /tmp/test_music3_out.wav <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"not a WAV: {b[:12]!r}"
fmt, channels, rate = struct.unpack("<HHI", b[20:28])
bits = struct.unpack("<H", b[34:36])[0]
assert fmt == 1 and bits == 16, (fmt, bits)
assert channels == 2, f"want stereo, got {channels}"
assert rate == 44100, f"want 44.1 kHz, got {rate}"
n_samples = (len(b) - 44) // (2 * channels)
dur = n_samples / rate
assert 0.5 <= dur <= 8.5, f"want (0.5, 8.5] s, got {dur:.2f} s"
assert any(b[44:44 + 4 * 44100]), "output is all-zero audio"
print(f"PASS: /v1/audio/music-generations -> {len(b)} byte WAV, {dur:.2f} s 44.1 kHz stereo")
PY
[ $? -eq 0 ] || exit 1
grep -q '\[music3\] AR stage' /tmp/test_music3_server.log || { echo "FAIL: no AR-stage engagement in log"; exit 1; }
echo "PASS: AR-stage engagement logged"

# 3b. Instrumental: NO lyrics field at all, just the flag. `is_instrumental` is
# a hosted-api convenience the open weights never had, so the flag becomes the
# `[Instrumental]` section tag MiniMax's model card lists. The bar is ACCEPTED +
# produces real audio, AND that the engine logs instrumental=true — a silently
# ignored flag would otherwise pass by generating a normal sung track.
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MUSIC_ID\",\"prompt\":\"slow lo-fi piano, rain, no vocals\",\"instrumental\":true,\"duration_seconds\":8,\"steps\":8,\"seed\":7}" \
  -o /tmp/test_music3_inst.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: instrumental gen http $code"; head -c 300 /tmp/test_music3_inst.wav; tail -20 /tmp/test_music3_server.log; exit 1; }
python3 - /tmp/test_music3_inst.wav <<'INSTPY'
import sys, struct
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"not a WAV: {b[:12]!r}"
channels, rate = struct.unpack("<HI", b[22:28])
n = (len(b) - 44) // (2 * channels)
assert channels == 2 and rate == 44100, (channels, rate)
assert n / rate >= 0.5, f"instrumental too short: {n / rate:.2f} s"
assert any(b[44:44 + 4 * 44100]), "instrumental output is all-zero audio"
print(f"PASS: instrumental (no lyrics field) -> {n / rate:.2f} s WAV")
INSTPY
[ $? -eq 0 ] || exit 1
grep -q 'instrumental=true' /tmp/test_music3_server.log \
  || { echo "FAIL: instrumental flag never reached the engine (no instrumental=true in log)"; exit 1; }
echo "PASS: instrumental=true logged — the flag is not a silent no-op"

# 4. Server survives the gen.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after music gen"; exit 1; }

# 5. Streaming: SSE progress (prefill/frames/diffuse/decode) + base64 complete.
cat > /tmp/test_music3_stream_req.json <<EOF
{"model":"$MUSIC_ID","prompt":"gentle acoustic folk guitar","lyrics":"[verse]\nsoft morning light","duration_seconds":6,"steps":8,"seed":7,"stream":true}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_music3_stream_req.json -o /tmp/test_music3_stream.txt -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream music gen http $code"; exit 1; }
grep -q '"stage":"frames"' /tmp/test_music3_stream.txt || { echo "FAIL: no AR-frame progress in stream"; exit 1; }
grep -q '"stage":"diffuse"' /tmp/test_music3_stream.txt || { echo "FAIL: no diffuse progress in stream"; exit 1; }
grep -q '"type":"complete"' /tmp/test_music3_stream.txt || { echo "FAIL: no complete event in stream"; exit 1; }
echo "PASS: streaming -> SSE frames/diffuse progress + complete event"

# 6. Coexistence with a chat model.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while music model resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with music model also resident"
fi

# 7. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$MUSIC_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$MUSIC_ID']
assert m and m[0]['state']=='unloaded', 'music model should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> music model unloaded (stub retained)')
"

echo "ALL PASS: MiniMax Music 3 gen (headless boot, load->gen->unload, WAV validity, named 400s, streaming, coexistence)"
