#!/usr/bin/env bash
# YuE2 song generation (`.audio` music backend): headless boot -> load the pack by
# absolute path -> /v1/models advertises "audio"+"music" -> the named-400 family
# (missing prompt/lyrics, bad cot, abc with cot off, ACE-Step-only fields, bad
# duration/steps/cfg, TTS endpoint on a music model) -> a short cot=off song as a
# valid 48 kHz stereo WAV -> a cot=full song over SSE carrying the ABC score it
# was rendered from, then the SAME score fed back as `abc` (planning skipped) ->
# unload. Asserts invariants (WAV shape, engagement lines, the score on the wire),
# never what the model chose to sing.
#
# Skips gracefully when no pack is present. Packs: ahmadw/YuE2-3B-MLX (one folder
# per width) copied flat into <root>/<org>/<name>, or ddalcu/YuE2-3B-MLX-Serve-8bit.
#
# Usage: YUE2_MODEL=<dir> [CHAT_MODEL=<dir>] ./tests/test_yue2_gen.sh [port]
set -uo pipefail
PORT="${1:-11439}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODEL="${YUE2_MODEL:-$(ls -d ~/.mlx-serve/models/ddalcu/YuE2-3B-MLX-Serve-8bit 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no YuE2 pack (set YUE2_MODEL to a converted dir)"; exit 0; }
[ -f "$MODEL/config.json" ] || { echo "SKIP: $MODEL has no config.json"; exit 0; }
[ -f "$MODEL/vae.safetensors" ] || { echo "SKIP: $MODEL is incomplete (no vae.safetensors marker)"; exit 0; }

TMP="$(mktemp -d)"
trap 'kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
# Scratch HOME: a developer's model-settings.json / providers.json must not reach the boot.
mkdir -p "$TMP/home" "$TMP/models"
HOME="$TMP/home" "$BIN" --serve --model-dir "$TMP/models" --port "$PORT" >"$TMP/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 "$TMP/server.log"; exit 1; }
  sleep 1
done

api() { curl -s -m 3600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
ID="$(basename "$MODEL")"

# 1. Load by absolute path -> ready with "audio" + "music".
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID' and x['state']=='ready']
assert m, 'YuE2 not ready: '+json.dumps(d)
caps=m[0].get('capabilities',[])
assert 'audio' in caps and 'music' in caps, f'want audio+music caps, got {caps}'
print('PASS: load-model by path -> ready, capabilities', caps)
" || { echo "FAIL: ready model missing audio/music capability"; exit 1; }

# 2. The 400 family, before anything expensive.
b400() { # label body [needle the message must contain]
  local code
  code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
    -d "$2" -o "$TMP/err.txt" -w "%{http_code}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat "$TMP/err.txt"; exit 1; }
  if [ -n "${3:-}" ]; then
    grep -q "$3" "$TMP/err.txt" || { echo "FAIL: $1 400 does not name '$3'"; cat "$TMP/err.txt"; exit 1; }
  fi
  echo "PASS: $1 -> 400"
}
LY='[Verse]\nla la la\n[Chorus]\nhey hey hey'
b400 "missing prompt" "{\"model\":\"$ID\",\"lyrics\":\"$LY\"}" prompt
b400 "missing lyrics" "{\"model\":\"$ID\",\"prompt\":\"pop\"}" lyrics
b400 "empty lyrics" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"  \"}" lyrics
b400 "bad cot" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"cot\":\"loud\"}" cot
b400 "abc with cot off" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"cot\":\"off\",\"abc\":\"X:1\"}" abc
b400 "empty abc" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"abc\":\" \"}" abc
b400 "duration 1" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"duration_seconds\":1}" duration
b400 "duration 999" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"duration_seconds\":999}" duration
b400 "steps 0" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"steps\":0}" steps
b400 "cfg 99" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",\"cfg_scale\":99}" cfg_scale
# Fields another backend owns are refused BY NAME, never ignored.
for f in 'bpm:"bpm":120' 'keyscale:"keyscale":"C major"' 'timesignature:"timesignature":"4/4"' \
         'vocal_language:"vocal_language":"en"' 'ref_audio:"ref_audio":"AAAA"' 'src_audio:"src_audio":"AAAA"' \
         'task:"task":"cover"' 'instrumental:"instrumental":true'; do
  name="${f%%:*}"; frag="${f#*:}"
  b400 "unsupported field $name" "{\"model\":\"$ID\",\"prompt\":\"pop\",\"lyrics\":\"$LY\",$frag}" "$name"
done
code=$(api /v1/audio/speech -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"input\":\"hello\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: /v1/audio/speech on a music model returned $code (want 400)"; exit 1; }
echo "PASS: /v1/audio/speech on a music model -> 400"

wavcheck() { # file min_s max_s label
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
lo, hi, label = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"not a WAV: {b[:12]!r}"
fmt, channels, rate = struct.unpack("<HHI", b[20:28])
bits = struct.unpack("<H", b[34:36])[0]
assert fmt == 1 and bits == 16 and channels == 2 and rate == 48000, (fmt, bits, channels, rate)
n = (len(b) - 44) // (2 * channels)
dur = n / rate
assert lo <= dur <= hi, f"want [{lo},{hi}] s, got {dur:.2f} s"
assert any(b[44 + 4 * 48000:44 + 8 * 48000]), "output is all-zero audio"
# A frame is 1920 samples less 64: the decoder's odd-stride crop, exact for any length.
assert (n + 64) % 1920 == 0, f"length {n} is not a whole number of frames"
print(f"PASS: {label} -> {dur:.2f} s 48 kHz stereo PCM16")
PY
}

# 3. A short cot=off song: the WAV is whole frames, 48 kHz stereo, not silent.
cat > "$TMP/off.json" <<EOF
{"model":"$ID","prompt":"English, pop, bright acoustic guitar, warm female vocal","lyrics":"$LY","cot":"off","duration_seconds":6,"steps":4,"seed":7}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @"$TMP/off.json" -o "$TMP/off.wav" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: cot=off gen http $code"; head -c 300 "$TMP/off.wav"; tail -20 "$TMP/server.log"; exit 1; }
wavcheck "$TMP/off.wav" 4 6.5 "cot=off song" || exit 1
grep -q '\[yue2\] semantic: .*cfg 1.01' "$TMP/server.log" || { echo "FAIL: cot=off did not run the CFG branch (cfg 1.01)"; exit 1; }
grep -q '\[yue2\] planning' "$TMP/server.log" && { echo "FAIL: cot=off planned a score"; exit 1; }
echo "PASS: cot=off -> no planning phase, CFG 1.01"

# 3b. Same request and seed twice: byte-identical (host sampler + noise are both seeded).
api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @"$TMP/off.json" -o "$TMP/off2.wav" >/dev/null
cmp -s "$TMP/off.wav" "$TMP/off2.wav" || { echo "FAIL: same seed produced different audio"; exit 1; }
echo "PASS: same seed -> identical WAV"

# 4. cot=full over SSE: planning progress, the score on the complete event.
cat > "$TMP/full.json" <<EOF
{"model":"$ID","prompt":"English, indie pop, acoustic guitar, soft drums, warm lead vocal","lyrics":"$LY","cot":"full","duration_seconds":8,"steps":8,"seed":7,"stream":true}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @"$TMP/full.json" -o "$TMP/full.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream gen http $code"; exit 1; }
for st in abc semantic nar decode; do
  grep -q "\"stage\":\"$st\"" "$TMP/full.txt" || { echo "FAIL: no '$st' progress in the stream"; exit 1; }
done
python3 - "$TMP/full.txt" "$TMP" <<'PY'
import sys, json, base64
evs = [json.loads(l[6:]) for l in open(sys.argv[1], encoding="utf-8").read().split("\n") if l.startswith("data: ")]
done = [e for e in evs if e.get("type") == "complete"]
assert len(done) == 1, f"want one complete event, got {len(done)}"
abc = done[0].get("abc", "")
assert abc.lstrip().startswith("X:"), f"complete event carries no ABC score: {abc[:60]!r}"
# AR phases have no known length: indeterminate progress, never a fake bar.
ar = [e for e in evs if e.get("type") == "progress" and e.get("stage") in ("abc", "semantic")]
assert ar and all(e["total"] == 0 for e in ar), "AR progress must be indeterminate (total 0)"
open(sys.argv[2] + "/score.abc", "w", encoding="utf-8").write(abc)
open(sys.argv[2] + "/full.wav", "wb").write(base64.b64decode(done[0]["data"]))
print(f"PASS: cot=full stream -> abc/semantic/nar/decode progress + complete event with a {len(abc)}-char ABC score")
PY
[ $? -eq 0 ] || exit 1
wavcheck "$TMP/full.wav" 3 8.5 "cot=full song" || exit 1

# 5. The score fed back as `abc`: planning is skipped and the song is still made.
python3 - "$TMP" "$ID" <<'PY'
import json, sys
tmp, mid = sys.argv[1], sys.argv[2]
abc = open(tmp + "/score.abc", encoding="utf-8").read()
body = {"model": mid, "prompt": "Jazz, piano, upright bass, brushed drums", "lyrics": "[Verse]\nla la la\n[Chorus]\nhey hey hey",
        "cot": "full", "abc": abc, "duration_seconds": 6, "steps": 4, "seed": 7}
json.dump(body, open(tmp + "/edited.json", "w"))
PY
mark=$(wc -l < "$TMP/server.log")
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @"$TMP/edited.json" -o "$TMP/edited.wav" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: abc-supplied gen http $code"; head -c 300 "$TMP/edited.wav"; exit 1; }
wavcheck "$TMP/edited.wav" 3 6.5 "song from a supplied score" || exit 1
tail -n +"$((mark + 1))" "$TMP/server.log" | grep -q '\[yue2\] planning' \
  && { echo "FAIL: a supplied score was re-planned"; exit 1; }
echo "PASS: abc supplied -> no planning phase"

# 6. Server alive, no MLX error, then coexist with a chat model.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died"; exit 1; }
grep -q '\[mlx\]' "$TMP/server.log" && { echo "FAIL: MLX error in the log"; grep '\[mlx\]' "$TMP/server.log" | head -3; exit 1; }
if [ -d "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while the music model was resident"; exit 1; }
  echo "PASS: chat streams ($TOK deltas) with the music model also resident"
fi

# 7. Unload -> the stub stays, unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ID']
assert m and m[0]['state']=='unloaded', 'model should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> unloaded (stub retained)')
"

echo "ALL PASS: YuE2 gen (load->gen->unload, named 400s, WAV frames, seeded repeat, planned and supplied scores)"
