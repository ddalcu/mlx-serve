#!/usr/bin/env bash
# test_tts_language.sh — Qwen3-TTS `language` on /v1/audio/speech: an undeclared name or a dialect is a named 400
# (plain and streamed), `auto` is byte-identical to no language, a declared language changes the audio.
#
#   TTS_MODEL=<Qwen3-TTS dir> ./tests/test_tts_language.sh [port]   # SKIPs without one
#
# Hermetic twins: the `language:` test in tts.zig and the plugin's language test on the tiny pack.
set -euo pipefail
cd "$(dirname "$0")/.."
source tests/lib/server.sh
source tests/_lib_models.sh
PORT="${1:-11379}"
BASE="http://127.0.0.1:$PORT"
MODEL="${TTS_MODEL:-$(find_model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 || true)}"
[ -n "$MODEL" ] || { echo "SKIP: no qwen3_tts model (set TTS_MODEL)"; exit 0; }
[ -d "$MODEL" ] || { echo "FAIL: TTS_MODEL points at a missing dir: $MODEL" >&2; exit 1; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
TMP=$(mktemp -d "$HOME/claude-tmp/test_tts_language.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$TMP/home"
HOME="$TMP/home" "$BIN" --model "$MODEL" --serve --port "$PORT" >"$TMP/server.log" 2>&1 &
SRV=$!; trap 'srv_stop "$SRV"; rm -rf "$TMP"' EXIT
srv_wait_health "$BASE" 180 0.5 "$SRV" || { echo "FAIL: server did not start"; tail -20 "$TMP/server.log"; exit 1; }

python3 - "$BASE" <<'PY'
import json, sys, urllib.error, urllib.request
url = sys.argv[1] + '/v1/audio/speech'
text = 'Entra Dalton. Pra ele, cada elemento tem seu tipo de átomo.'
def post(body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()

for language in ('klingon', 'beijing_dialect', 2071):
    for stream in (False, True):
        status, body = post({'input': text, 'language': language, 'stream': stream})
        assert status == 400, f'language={language} stream={stream} -> {status} (want 400)'
        assert 'language' in json.loads(body)['error']['message'], body
        print(f'PASS: language={language} stream={stream} -> named HTTP 400')

status, plain = post({'input': text})
assert status == 200 and plain[:4] == b'RIFF', status
status, auto = post({'input': text, 'language': 'auto'})
assert status == 200 and auto == plain, 'language=auto must be the plain path'
print('PASS: language=auto is byte-identical to no language')

status, pt = post({'input': text, 'language': 'Portuguese'})
assert status == 200 and pt[:4] == b'RIFF' and len(pt) > 40000, (status, len(pt))
assert pt != plain, 'language=Portuguese produced the auto audio: the language did not reach the codec prefix'
print(f'PASS: language=Portuguese conditions synthesis ({len(pt)} byte WAV)')
PY
