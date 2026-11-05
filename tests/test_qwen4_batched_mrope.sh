#!/usr/bin/env bash
# qwen4_exp: image-bearing slots decode in one batched tick. Three concurrent chats, each with images, two turns.
# Asserts every request answers, no decode tick fails, and the batched GDN tick engages.
#   QWEN4_MODEL=<pack dir> ./tests/test_qwen4_batched_mrope.sh [port]
source "$(dirname "$0")/_lib_models.sh"
set -u
MODEL="${QWEN4_MODEL:-$(find_model ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw || echo "$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw")}"
PORT="${1:-11418}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
DIR="${OUT_DIR:-$HOME/claude-tmp/qwen4-batched-mrope}"
mkdir -p "$DIR"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no pack at $MODEL"; exit 0; }
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }

LOG="$DIR/server.log"
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info --max-concurrent 3 \
  --prefix-cache-entries 4 > "$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT INT TERM
for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$LOG" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }; sleep 2; done

python3 - "$PORT" > "$DIR/statuses.txt" <<'PY'
import base64, json, struct, sys, threading, urllib.request, zlib
port = int(sys.argv[1])
def png(seed, w=320, h=240):
    rows = b''.join(b'\x00' + b''.join(bytes(((x * seed + y * 7) % 256, (x + y * seed) % 256, seed * 31 % 256)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()
filler = 'The archivist catalogued the shelves in the long hall, one ledger per row. ' * 300
def chat(k, out):
    content = [{'type': 'text', 'text': filler}]
    for j in range(2 + k % 2):
        content += [{'type': 'image_url', 'image_url': {'url': png(3 + 10 * k + j)}}, {'type': 'text', 'text': f'Panel {j}.'}]
    content.append({'type': 'text', 'text': 'Describe each panel in two sentences.'})
    msgs = [{'role': 'user', 'content': content}]
    for turn in range(2):
        body = json.dumps({'messages': msgs, 'max_tokens': 200, 'temperature': 0.7, 'enable_thinking': False}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'content-type': 'application/json'}), timeout=1800) as r:
                reply = json.load(r)['choices'][0]['message']['content']; out.append(200)
        except urllib.error.HTTPError as e:
            out.append(e.code); return
        msgs += [{'role': 'assistant', 'content': reply}, {'role': 'user', 'content': 'Now compare the panels.'}]
out = []
th = [threading.Thread(target=chat, args=(k, out)) for k in range(3)]
for t in th: t.start()
for t in th: t.join()
print(' '.join(map(str, sorted(out))))
PY
check "every request answers 200" "$(cat "$DIR/statuses.txt")" "200 200 200 200 200 200"
check "no decode tick fails" "$(grep -c 'decode tick failed' "$LOG")" "0"
check "batched GDN tick engaged" "$(grep -q '\[batched\] gdn batched decode engaged' "$LOG" && echo yes)" "yes"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
