#!/usr/bin/env bash
# HeartMuLa song generation (`.audio` music backend): headless boot -> load the pack by absolute path ->
# /v1/models advertises "audio"+"music" -> the named-400 family (missing prompt/lyrics, bad duration/steps/cfg/
# temperature/top_k, fields other backends own, a prompt past the backbone's positions) -> a short song as a valid
# 48 kHz stereo WAV with the frame count on the log -> the same seed twice is byte-identical -> a streamed song
# reports `song` then `nar` progress -> unload. Asserts invariants (WAV shape, length within the cap, engagement
# lines), never what the model chose to sing.
#
# Skips when no pack is present: ddalcu/HeartMuLa-3B-MLX-Serve-8bit (scripts/convert_heartmula_weights.py).
#
# Usage: HEARTMULA_MODEL=<dir> ./tests/test_heartmula_gen.sh [port]
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11441}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODEL="${HEARTMULA_MODEL:-$(ls -d ~/.mlx-serve/models/ddalcu/HeartMuLa-3B-MLX-Serve-8bit 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no HeartMuLa pack (set HEARTMULA_MODEL to a converted dir)"; exit 0; }
[ -f "$MODEL/codec.safetensors" ] || { echo "SKIP: $MODEL is incomplete (no codec.safetensors marker)"; exit 0; }

TMP="$(mktemp -d)"
trap 'kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
mkdir -p "$TMP/home" "$TMP/models"
HOME="$TMP/home" "$BIN" --serve --model-dir "$TMP/models" --port "$PORT" >"$TMP/server.log" 2>&1 &
SRV=$!
srv_wait_health "http://127.0.0.1:$PORT" 60 1 "$SRV" || kill -0 "$SRV" 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 "$TMP/server.log"; exit 1; }

api() { curl -s -m 3600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
ID="$(basename "$MODEL")"

api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$ID' and x['state']=='ready']
assert m, 'HeartMuLa not ready'
caps=m[0].get('capabilities',[])
assert 'audio' in caps and 'music' in caps, f'want audio+music caps, got {caps}'
print('PASS: load-model by path -> ready, capabilities', caps)
" || { echo "FAIL: ready model missing audio/music capability"; tail -20 "$TMP/server.log"; exit 1; }

b400() { # label body needle
  local code
  code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' -d "$2" -o "$TMP/err.txt" -w "%{http_code}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat "$TMP/err.txt"; exit 1; }
  grep -q "$3" "$TMP/err.txt" || { echo "FAIL: $1 400 does not name '$3'"; cat "$TMP/err.txt"; exit 1; }
  echo "PASS: $1 -> 400"
}
LY='[Verse]\nneon lights across the bay\n[Chorus]\nhold on tonight'
B="\"model\":\"$ID\",\"prompt\":\"pop, piano\",\"lyrics\":\"$LY\""
b400 "missing prompt" "{\"model\":\"$ID\",\"lyrics\":\"$LY\"}" prompt
b400 "missing lyrics" "{\"model\":\"$ID\",\"prompt\":\"pop\"}" lyrics
b400 "duration 4" "{$B,\"duration_seconds\":4}" duration
b400 "duration 999" "{$B,\"duration_seconds\":999}" duration
b400 "steps 2" "{$B,\"steps\":2}" steps
b400 "cfg 0.5" "{$B,\"cfg_scale\":0.5}" cfg_scale
b400 "temperature 0" "{$B,\"temperature\":0}" temperature
b400 "top_k 0" "{$B,\"top_k\":0}" top_k
for f in 'bpm:"bpm":120' 'keyscale:"keyscale":"C major"' 'timesignature:"timesignature":"4/4"' \
         'vocal_language:"vocal_language":"en"' 'ref_audio:"ref_audio":"AAAA"' 'src_audio:"src_audio":"AAAA"' \
         'task:"task":"cover"' 'instrumental:"instrumental":true'; do
  b400 "unsupported field ${f%%:*}" "{$B,${f#*:}}" "${f%%:*}"
done
python3 -c "import json;print(json.dumps({'model':'$ID','prompt':'pop','lyrics':'la '*9000,'duration_seconds':360}))" > "$TMP/long.json"
b400 "prompt past the backbone positions" @"$TMP/long.json" "8192 positions"

wavcheck() { # file max_s label
  python3 - "$1" "$2" "$3" <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
hi, label = float(sys.argv[2]), sys.argv[3]
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"not a WAV: {b[:12]!r}"
fmt, channels, rate = struct.unpack("<HHI", b[20:28])
assert fmt == 1 and channels == 2 and rate == 48000, (fmt, channels, rate)
dur = (len(b) - 44) / 4 / rate
assert 0.08 <= dur <= hi, f"want (0, {hi}] s, got {dur:.2f} s"
assert any(b[44:]), "output is all-zero audio"
print(f"PASS: {label} -> {dur:.2f} s 48 kHz stereo PCM16")
PY
}

cat > "$TMP/song.json" <<EOF
{$B,"duration_seconds":10,"steps":10,"seed":7}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' -d @"$TMP/song.json" -o "$TMP/a.wav" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: gen http $code"; head -c 300 "$TMP/a.wav"; tail -20 "$TMP/server.log"; exit 1; }
wavcheck "$TMP/a.wav" 10.1 "10 s cap song" || exit 1
grep -q '\[heartmula\] [0-9]* frames' "$TMP/server.log" || { echo "FAIL: no frame-count line on the log"; exit 1; }
echo "PASS: $(grep -o '\[heartmula\] [0-9]* frames ([0-9.]*s)' "$TMP/server.log" | tail -1)"

api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' -d @"$TMP/song.json" -o "$TMP/b.wav" >/dev/null
cmp -s "$TMP/a.wav" "$TMP/b.wav" || { echo "FAIL: same seed produced different audio"; exit 1; }
echo "PASS: same seed -> identical WAV"

# Streamed: composing reports seconds with no total (the model decides the length), then codec steps.
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d "{$B,\"duration_seconds\":5,\"steps\":4,\"seed\":7,\"stream\":true}" -o "$TMP/stream.txt" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream gen http $code"; exit 1; }
grep -q '"stage":"song","step":[0-9]*,"total":0' "$TMP/stream.txt" || { echo "FAIL: no indeterminate 'song' progress"; grep -m3 progress "$TMP/stream.txt"; exit 1; }
grep -q '"stage":"nar"' "$TMP/stream.txt" || { echo "FAIL: no 'nar' (codec) progress"; exit 1; }
grep -q '"type":"complete"' "$TMP/stream.txt" || { echo "FAIL: no complete event"; exit 1; }
echo "PASS: stream -> song (seconds, total 0) then nar progress + complete"

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died"; exit 1; }
grep -q '\[mlx\]' "$TMP/server.log" && { echo "FAIL: MLX error in the log"; grep '\[mlx\]' "$TMP/server.log" | head -3; exit 1; }
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$ID']
assert m and m[0]['state']=='unloaded', 'model should be unloaded'
print('PASS: unload-model -> unloaded (stub retained)')
" || exit 1
echo "ALL PASS: HeartMuLa gen (load->gen->unload, named 400s, WAV shape, seeded repeat)"
