#!/bin/bash
# test_prefill_progress.sh — `return_progress` streams `prompt_progress` chunks while a long
# prefill runs; a request without the flag never sees one.
#
#   ./tests/test_prefill_progress.sh [model_dir] [port]
#
# A long prompt and a narrow --prefill-chunk keep the prefill alive for several poll intervals,
# so at least one progress chunk lands while the prefill is still running.
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
PORT="${2:-11534}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }

WORK=$(mktemp -d "$HOME/claude-tmp/prefill-progress.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$WORK/home"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 0.5
HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --prefill-chunk 2048 --log-level info > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
trap 'srv_stop $SERVER_PID' EXIT
srv_wait_health "$BASE" 120 1 || { echo "FAIL: server did not start"; tail -20 "$WORK/server.log"; exit 1; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }

python3 -c "print('apple pear plum cherry melon apricot fig date lemon ' * 2600)" > "$WORK/filler.txt"

stream() { # <outfile> <return_progress true|false>
    python3 - "$BASE" "$WORK/filler.txt" "$2" > "$1" <<'PYEOF'
import json, sys, urllib.request
base, filler_path, return_progress = sys.argv[1], sys.argv[2], sys.argv[3]
filler = open(filler_path).read()
body = {"model": "m", "stream": True, "max_tokens": 8,
        "return_progress": return_progress == "true",
        "messages": [{"role": "user", "content": "Summarize in one word. " + filler}]}
r = urllib.request.urlopen(urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900)
for raw in r:
    line = raw.decode().strip()
    if line.startswith("data: ") and line != "data: [DONE]":
        try:
            print(json.dumps(json.loads(line[6:])))
        except Exception:
            pass
PYEOF
}

collect() { # <stream file> → the prompt_progress objects, as JSON lines
    python3 -c "
import json, sys
for line in open('$1'):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if 'prompt_progress' in d:
        print(json.dumps(d['prompt_progress']))
"
}

echo "=== with return_progress ==="
stream "$WORK/on.stream" true
PROG=$(collect "$WORK/on.stream")
N=$(echo -n "$PROG" | grep -c . || true)
if [[ "$N" -ge 1 ]]; then ok "at least one progress chunk ($N)"; else bad "no prompt_progress chunk in the stream" "$(head -c 300 "$WORK/on.stream")"; fi

if [[ "$N" -ge 1 ]]; then
    VERDICT=$(echo "$PROG" | python3 -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin]
bad = []
prev = -1
for p in rows:
    if not all(k in p for k in ('total', 'cache', 'processed', 'time_ms')):
        bad.append('missing fields: ' + json.dumps(p))
    if p['processed'] < prev:
        bad.append('processed moved backwards: ' + json.dumps(p))
    prev = p['processed']
last = rows[-1]
# The poll can miss the final chunk boundary (the first token may arrive before
# the next poll), so the last frame may sit one chunk short of the prompt's end.
if last['processed'] < last['total'] - 2048:
    bad.append('last processed well short of total: ' + json.dumps(last))
if last['total'] < 1024:
    bad.append('prompt unexpectedly small: ' + str(last['total']))
print('OK' if not bad else '; '.join(bad))
")
    if [[ "$VERDICT" == "OK" ]]; then ok "progress fields sane, reaches the prompt's end"; else bad "progress stream malformed" "$VERDICT"; fi
fi

echo "=== without return_progress ==="
stream "$WORK/off.stream" false
OFF=$(collect "$WORK/off.stream")
if [[ -z "$OFF" ]]; then ok "no progress chunk without the flag"; else bad "progress leaked without the flag" "$OFF"; fi

echo
echo "PASS=$PASS FAIL=$FAIL"
[[ $FAIL -eq 0 ]]
