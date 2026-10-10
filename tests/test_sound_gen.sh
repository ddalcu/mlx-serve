#!/usr/bin/env bash
# Stable Audio 3 text-to-audio on the ONE main server: the OFFICIAL
# stabilityai/stable-audio-3-small-* repo (no config.json, only
# model_config.json) discovered under --model-dir -> stub advertises
# "audio"+"sound" -> load by id -> the 400 family (missing/empty prompt,
# duration/steps bounds, negative or fractional numbers, speech/music endpoints
# refuse the backend) ->
# POST /v1/audio/sound-generations -> a 44.1 kHz stereo PCM16 WAV of EXACTLY
# the requested length -> a seed reproduces the same bytes, another seed does
# not -> SSE diffuse progress + base64 complete -> unload.
#
# Skips when the repo is absent. Download with:
#   hf download ddalcu/Stable-Audio-3-Small-SFX-MLX-Serve --local-dir ~/.mlx-serve/models/stabilityai/stable-audio-3-small-sfx
#
# Usage: SA3_MODEL=<dir> ./tests/test_sound_gen.sh [port]
set -uo pipefail
PORT="${1:-11441}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

SA3="${SA3_MODEL:-$HOME/.mlx-serve/models/stabilityai/stable-audio-3-small-sfx}"
[ -f "$SA3/model_config.json" ] || { echo "SKIP: no Stable Audio 3 repo at $SA3 (set SA3_MODEL)"; exit 0; }
[ -f "$SA3/t5gemma-b-b-ul2/model.safetensors" ] || { echo "SKIP: $SA3 is incomplete (no t5gemma-b-b-ul2/model.safetensors)"; exit 0; }

# A private root holding only this repo, so discovery is the thing under test.
TMP="$(mktemp -d)"
mkdir -p "$TMP/models/stabilityai"
ln -s "$SA3" "$TMP/models/stabilityai/$(basename "$SA3")"
ID="stabilityai/$(basename "$SA3")"
LOG="$TMP/server.log"
"$BIN" --serve --model-dir "$TMP/models" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done

api() { curl -s -m 600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
post() { api "$1" -X POST -H 'Content-Type: application/json' -d "$2" "${@:3}"; }

caps() { # state
  api /v1/models | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$ID']
assert m, 'not discovered'
assert m[0]['state']=='$1', m[0]['state']
c=m[0].get('capabilities',[])
assert 'audio' in c and 'sound' in c and 'music' not in c, c
print(c)"
}

# 1. Discovery from model_config.json alone: the stub already says what it is.
c=$(caps unloaded) || { echo "FAIL: stub not discovered with audio+sound"; exit 1; }
echo "PASS: discovered as a stub, capabilities $c"
post /v1/load-model "{\"model\":\"$ID\"}" >/dev/null
c=$(caps ready) || { echo "FAIL: loaded model lacks audio+sound"; tail -20 "$LOG"; exit 1; }
echo "PASS: load by id -> ready, capabilities $c"

# 2. The 400 family, before any generation.
b400() { # label path body needle
  local code
  code=$(post "$2" "$3" -o "$TMP/err.txt" -w "%{http_code}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat "$TMP/err.txt"; exit 1; }
  grep -q "$4" "$TMP/err.txt" || { echo "FAIL: $1 400 does not name '$4'"; cat "$TMP/err.txt"; exit 1; }
  echo "PASS: $1 -> 400"
}
S=/v1/audio/sound-generations
b400 "missing prompt" $S "{\"model\":\"$ID\"}" prompt
b400 "empty prompt" $S "{\"model\":\"$ID\",\"prompt\":\"\"}" prompt
b400 "duration 0" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"duration_seconds\":0}" duration_seconds
b400 "duration 121" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"duration_seconds\":121}" duration_seconds
b400 "steps 0" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"steps\":0}" steps
b400 "steps 51" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"steps\":51}" steps
b400 "steps -5" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"steps\":-5}" steps
b400 "steps 4.9" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"steps\":4.9}" steps
b400 "seed -1" $S "{\"model\":\"$ID\",\"prompt\":\"rain\",\"seed\":-1}" seed
b400 "speech endpoint" /v1/audio/speech "{\"model\":\"$ID\",\"input\":\"hello\"}" sound-generations
b400 "music endpoint" /v1/audio/music-generations "{\"model\":\"$ID\",\"prompt\":\"jazz\"}" sound-generations

# 3. Generate: exact length, stereo 44.1 kHz, not silent.
gen() { # out body
  local code
  code=$(post $S "$2" -o "$1" -w "%{http_code}")
  [ "$code" = "200" ] || { echo "FAIL: sound gen http $code"; head -c 300 "$1"; tail -20 "$LOG"; exit 1; }
}
gen "$TMP/a.wav" "{\"model\":\"$ID\",\"prompt\":\"Dog barking next to a waterfall\",\"duration_seconds\":3.5,\"seed\":7}"
python3 - "$TMP/a.wav" <<'PY' || exit 1
import sys, struct, array
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", b[:12]
fmt, ch, rate = struct.unpack("<HHI", b[20:28])
bits = struct.unpack("<H", b[34:36])[0]
assert (fmt, ch, rate, bits) == (1, 2, 44100, 16), (fmt, ch, rate, bits)
n = (len(b) - 44) // 4
assert n == round(3.5 * 44100), f"want {round(3.5*44100)} frames, got {n}"
pcm = array.array("h", b[44:])
rms = (sum(v * v for v in pcm) / len(pcm)) ** 0.5 / 32768
assert rms > 0.005, f"near-silent output (rms {rms:.4f})"
print(f"PASS: 3.5 s request -> {n} stereo frames at 44.1 kHz, rms {rms:.3f}")
PY
grep -q '\[sa3\] 3.50s -> 108 latents' "$LOG" || { echo "FAIL: no [sa3] engagement line"; exit 1; }
echo "PASS: engine engagement logged"

# 4. A seed reproduces the same bytes; another seed is another sound.
gen "$TMP/b.wav" "{\"model\":\"$ID\",\"prompt\":\"Dog barking next to a waterfall\",\"duration_seconds\":3.5,\"seed\":7}"
gen "$TMP/c.wav" "{\"model\":\"$ID\",\"prompt\":\"Dog barking next to a waterfall\",\"duration_seconds\":3.5,\"seed\":8}"
cmp -s "$TMP/a.wav" "$TMP/b.wav" || { echo "FAIL: same seed gave different audio"; exit 1; }
cmp -s "$TMP/a.wav" "$TMP/c.wav" && { echo "FAIL: a different seed gave identical audio"; exit 1; }
echo "PASS: seed 7 twice is byte-identical, seed 8 differs"

# 5. Streaming: one diffuse event per step, then the WAV.
code=$(post $S "{\"model\":\"$ID\",\"prompt\":\"door creak\",\"duration_seconds\":1,\"steps\":4,\"stream\":true}" -o "$TMP/s.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream http $code"; exit 1; }
n=$(grep -c '"stage":"diffuse"' "$TMP/s.txt")
[ "$n" = "4" ] || { echo "FAIL: want 4 diffuse progress events, got $n"; cat "$TMP/s.txt" | head -5; exit 1; }
grep -q '"type":"complete","format":"wav"' "$TMP/s.txt" || { echo "FAIL: no complete event"; exit 1; }
echo "PASS: streaming -> 4 diffuse events + base64 WAV"

# 6. Unload.
post /v1/unload-model "{\"model\":\"$ID\"}" >/dev/null
caps unloaded >/dev/null || { echo "FAIL: not unloaded"; exit 1; }
echo "PASS: unload -> stub retained"
grep -q '\[mlx\]' "$LOG" && { echo "FAIL: MLX error in log"; grep '\[mlx\]' "$LOG" | head -3; exit 1; }
echo "ALL PASS: Stable Audio 3 sound generation"
