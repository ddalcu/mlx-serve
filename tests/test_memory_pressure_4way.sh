#!/bin/bash
# Four long prompts at once under a LOW guard ceiling: each is served or refused by name,
# a request that does not fit beside live ones WAITS, and the server is alive at the end.
#   MEMPRESS_TEST_MODEL=/path ./tests/test_memory_pressure_4way.sh [port]
set -u
PORT="${1:-11493}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${MEMPRESS_TEST_MODEL:-$HOME/.mlx-serve/models/lmstudio-community/Qwen3.5-2B-MLX-4bit}"
CEILING_MB="${MEMPRESS_CEILING_MB:-2600}"
WORK="$(mktemp -d)"; SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
[ -x "$BINARY" ] || { echo "[fail] build first"; exit 1; }
[ -d "$MODEL" ] || { echo "[skip] no model at $MODEL"; exit 0; }

MLX_SERVE_GPU_CEILING_MB=$CEILING_MB "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 65536 > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 90); do curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

python3 - "$PORT" > "$WORK/out.txt" <<'PY'
import json, sys, threading, urllib.request, urllib.error, uuid
url = f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions"
res = []
def run(i):
    text = uuid.uuid4().hex + (" The archive contains routine project notes and implementation details" * 2600) + "\nWrite the integers from 1 to 300, one per line."
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 400, "temperature": 0, "reasoning_effort": "high"}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=1200)
        res.append((r.status, ""))
    except urllib.error.HTTPError as e:
        res.append((e.code, e.read().decode(errors="replace")))
    except Exception as e:
        res.append((-1, repr(e)))
ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
for code, msg in res: print(code, msg[:160].replace("\n", " "))
PY
cat "$WORK/out.txt"; grep "\[admission\]" "$WORK/server.log" | head -6
FAIL=0
grep -qv "^200 \|^400 .*\(memory\|needs\)" "$WORK/out.txt" && { echo "[fail] a request was neither served nor refused by name"; FAIL=1; }
grep -q "^200 " "$WORK/out.txt" || { echo "[fail] nothing was served: raise MEMPRESS_CEILING_MB"; FAIL=1; }
grep -q "\[admission\] held" "$WORK/server.log" || { echo "[fail] no request waited: the ceiling never bound"; FAIL=1; }
curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[fail] server dead"; FAIL=1; }
grep "\[admission\] held\|\[diag\]" "$WORK/server.log" | head -5
[ $FAIL = 0 ] && echo "[pass] memory pressure 4-way" || { tail -20 "$WORK/server.log"; exit 1; }
