#!/usr/bin/env bash
# test_ming_tts.sh — Ming-3.0-tts on /v1/audio/speech: voices, a seed repeats a take, cloning, the French path, long text,
# named 400s, and a client that leaves mid-job.
#
#   MING_MODEL=<Ming-3.0-tts-0.5B dir, bf16 or quantized> ./tests/test_ming_tts.sh [port]   # SKIPs without one
set -euo pipefail
cd "$(dirname "$0")/.."
source tests/lib/server.sh
source tests/_lib_models.sh
PORT="${1:-11381}"
BASE="http://127.0.0.1:$PORT"
MODEL="${MING_MODEL:-$(find_model mlx-community/Ming-3.0-tts-0.5B-8bit || find_model inclusionAI/Ming-3.0-tts-0.5b || true)}"
[ -n "$MODEL" ] || { echo "SKIP: no Ming-3.0-tts-0.5b pack (set MING_MODEL)"; exit 0; }
[ -d "$MODEL" ] || { echo "FAIL: MING_MODEL points at a missing dir: $MODEL" >&2; exit 1; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
TMP=$(mktemp -d "$HOME/claude-tmp/test_ming_tts.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$TMP/home"
HOME="$TMP/home" "$BIN" --model "$MODEL" --serve --port "$PORT" >"$TMP/server.log" 2>&1 &
SRV=$!; trap 'srv_stop "$SRV"; rm -rf "$TMP"' EXIT
srv_wait_health "$BASE" 180 0.5 "$SRV" || { echo "FAIL: server did not start"; tail -20 "$TMP/server.log"; exit 1; }

python3 - "$BASE" <<'PY'
import base64, io, json, socket, sys, urllib.error, urllib.parse, urllib.request, wave
base = sys.argv[1]
url = base + '/v1/audio/speech'
def post(body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
def seconds(b):
    w = wave.open(io.BytesIO(b))
    assert w.getframerate() == 44100 and w.getnchannels() == 1
    return w.getnframes() / 44100

en = 'The garden stays open until sunset, and visitors leave by the north gate.'
status, take = post({'input': en, 'voice': 'UK_George', 'seed': 3})
assert status == 200 and take[:4] == b'RIFF' and 2 < seconds(take) < 12, (status, len(take))
status, again = post({'input': en, 'voice': 'UK_George', 'seed': 3})
assert status == 200 and again == take, 'the same seed must repeat the take'
status, other = post({'input': en, 'voice': 'UK_George', 'seed': 4})
assert status == 200 and other != take, 'another seed must change the take'
print('PASS: built-in voice, a seed repeats it')

ref = base64.b64encode(take).decode()
status, clone = post({'input': 'A short cloned sentence.', 'ref_audio': ref, 'ref_text': en, 'seed': 1})
assert status == 200 and seconds(clone) > 0.5, status
print('PASS: clone')
status, fr = post({'input': 'En 2024, le billet coûtait 12,50 €. C\'est l\'été.', 'ref_audio': ref, 'ref_text': en, 'seed': 1})
assert status == 200 and seconds(fr) > 1, status
print('PASS: French path')
status, designed = post({'input': en, 'instruction': 'A calm, low female voice.', 'seed': 2})
assert status == 200 and designed != take, status
status, emotion = post({'input': en, 'task': 'EMOTION', 'ref_audio': ref, 'controls': {'emotion': '高兴'}, 'seed': 2})
assert status == 200, (status, emotion[:200])
print('PASS: voice design, a task with controls')

status, long_take = post({'input': (en + '\n\n') * 3, 'voice': 'UK_George', 'seed': 3})
assert status == 200 and seconds(long_take) > 2.2 * seconds(take), (status, seconds(long_take))
print('PASS: long text joins its fragments')

for body, needle in (
    ({'input': 'hi', 'voice': 'alloy'}, 'voice'),
    ({'input': 'hi', 'warm_only': True}, 'warm_only'),
    ({'input': 'hi', 'ref_audio': '!!'}, 'ref_audio'),
    ({'input': 'hi', 'ref_text': 'x'}, 'ref_text'),
    ({'input': 'hi', 'task': 'NOPE'}, 'task'),
    ({'input': 'hi', 'task': 'BASIC'}, 'BASIC'),
    ({'input': 'hi', 'speed': 2}, 'speed'),
    ({'input': 'x' * 50001}, 'limit'),
):
    status, reply = post(body)
    assert status == 400 and needle in json.loads(reply)['error']['message'], (body.keys(), status, reply[:200])
print('PASS: named 400s')

# A client that leaves stops the job at its next fragment: the next request answers well before the long job would end.
parsed = urllib.parse.urlparse(url)
payload = json.dumps({'input': (en + ' ') * 40, 'voice': 'UK_George'}).encode()
s = socket.create_connection((parsed.hostname, parsed.port))
s.sendall(b'POST /v1/audio/speech HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(payload) + payload)
import time
time.sleep(2)
s.close()
start = time.time()
status, _ = post({'input': 'Still here.', 'voice': 'UK_George'})
assert status == 200 and time.time() - start < 20, (status, time.time() - start)
print(f'PASS: a client that leaves frees the GPU ({time.time() - start:.1f}s to the next answer)')
PY
