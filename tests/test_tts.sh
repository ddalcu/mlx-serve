#!/usr/bin/env bash
# Native Qwen3-TTS /v1/audio/speech endpoint smoke test.
# Usage: TTS_MODEL=<dir> ./tests/test_tts.sh [port]
set -euo pipefail
PORT="${1:-11377}"
BASE="http://localhost":"$PORT"
MODEL="${TTS_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16/snapshots/* 2>/dev/null | head -1 || true)}"
[ -n "$MODEL" ] || { echo "SKIP: no qwen3_tts model (set TTS_MODEL)"; exit 0; }
[ -d "$MODEL" ] || { echo "FAIL: TTS_MODEL points at a missing dir: $MODEL" >&2; exit 1; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/test_tts.XXXXXX")
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$TMP/server.log" 2>&1 &
SRV=$!; trap 'kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true; rm -rf "$TMP"' EXIT
for i in $(seq 1 90); do grep -q "Server listening" "$TMP/server.log" && break; sleep 1; done
grep -q "Server listening" "$TMP/server.log" || { echo "FAIL: server did not start"; cat "$TMP/server.log"; exit 1; }

# Bad references must fail before synthesis or SSE headers, never fall back to a plain voice.
for stream in false true; do
  for ref in '!!!!' 'bm90IGEgd2F2'; do
    code=$(curl -s -X POST "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
      -d "{\"input\":\"Hello.\",\"ref_audio\":\"$ref\",\"stream\":$stream}" -o "$TMP/error.json" -w "%{http_code}")
    [ "$code" = "400" ] || { echo "FAIL: invalid ref_audio stream=$stream -> $code (want 400)"; exit 1; }
    python3 - "$TMP/error.json" <<'PY'
import json, sys
assert 'ref_audio' in json.load(open(sys.argv[1]))['error']['message']
PY
    echo "PASS: invalid ref_audio stream=$stream -> named HTTP 400"
  done
done

code=$(curl -s -X POST "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"tts","input":"This is a native server test."}' -o "$TMP/speech.wav" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; exit 1; }
[ "$(head -c 4 "$TMP/speech.wav")" = "RIFF" ] || { echo "FAIL: not a WAV"; exit 1; }
sz=$(wc -c < "$TMP/speech.wav"); [ "$sz" -gt 40000 ] || { echo "FAIL: WAV too small ($sz)"; exit 1; }
echo "PASS: /v1/audio/speech -> $sz byte WAV"

code=$(curl -s -X POST "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"input":"Hello.","ref_audio":""}' -o "$TMP/empty-ref.wav" -w "%{http_code}")
[ "$code" = "200" ] && [ "$(head -c 4 "$TMP/empty-ref.wav")" = "RIFF" ] || { echo "FAIL: empty ref_audio changed plain synthesis"; exit 1; }
echo "PASS: empty ref_audio preserves plain synthesis"

# CustomVoice checkpoints cannot clone even a valid reference WAV.
python3 - "$BASE" "$TMP/speech.wav" <<'PY'
import base64, json, sys, urllib.error, urllib.request
url = sys.argv[1] + '/v1/audio/speech'
ref = base64.b64encode(open(sys.argv[2], 'rb').read()).decode()
def post(body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
status, body = post({'warm_only': True, 'ref_audio': ref})
if status == 200:
    print('SKIP: no-encoder rejection (this model supports cloning)')
else:
    assert status == 400 and 'no speaker encoder' in json.loads(body)['error']['message'], (status, body)
    for stream in (False, True):
        status, body = post({'input': 'Hello.', 'ref_audio': ref, 'stream': stream})
        assert status == 400, f'valid ref_audio without encoder stream={stream} -> {status} (want 400)'
        assert 'ref_audio' in json.loads(body)['error']['message']
        print(f'PASS: valid ref_audio without encoder stream={stream} -> named HTTP 400')
PY

# SSE streaming: indeterminate per-frame progress events, then a complete event
# carrying the WAV (audio length is model-determined, so total=0).
SSE="$TMP/speech.sse"
curl -sN -X POST "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"tts","input":"This is a longer sentence so the streaming progress updates over several frames.","stream":true}' >"$SSE"
python3 - "$SSE" <<'PY'
import sys, json, base64
prog = 0; complete = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev["type"] == "progress": prog += 1; assert {"stage","step","total"} <= set(ev)
    elif ev["type"] == "complete": complete = ev
assert prog >= 1, f"expected progress events, got {prog}"
assert complete is not None and complete.get("format") == "wav", "no wav complete event"
wav = base64.b64decode(complete["data"])
assert wav[:4] == b"RIFF", "complete WAV bad"
print(f"PASS: SSE stream -> {prog} progress events + complete WAV ({len(wav)} bytes)")
PY
