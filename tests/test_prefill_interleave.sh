#!/bin/bash
# test_prefill_interleave.sh — a cold prefill does not stall concurrent decode streams for its whole duration. A
# stream + an ~8k-token cold prefill at --prefill-chunk 1024: [1] the server logs `[interleave] engaged`. Two arms
# against a ~7-chunk prefill, share 0 vs --prefill-decode-share 0.5: [2] the share arm's hosted decode is over 1.5x
# share 0's fraction of the prefill wall time (`[interleave] prefill X ms, hosted decode Y ms`).
#
#   INTERLEAVE_TEST_MODEL=<small chat model> ./tests/test_prefill_interleave.sh [port]   # SKIPs without one

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11487}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${INTERLEAVE_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E2B-it-qat-4bit}"
WORK="$(mktemp -d)"
SERVER_PID=""

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "[skip] no model at $MODEL (set INTERLEAVE_TEST_MODEL)"
    exit 0
fi

cat > "$WORK/probe.py" <<'PYEOF'
import hashlib, json, sys, threading, time, urllib.request

PORT = int(sys.argv[1])
B_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def stream(payload, stamps, parts):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                obj = json.loads(line[6:])
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {}).get("content")
                    if d:
                        stamps.append(time.monotonic())
                        parts.append(d)

a_st, a_tx, b_st = [], [], []
a = {"model": "mlx-serve", "temperature": 0.0, "max_tokens": 500, "stream": True,
     "messages": [{"role": "user", "content": "Count from 1 to 200, one number per line."}]}
words = " ".join(f"w{i}" for i in range(B_WORDS))
b = {"model": "mlx-serve", "temperature": 0.0, "max_tokens": 4, "stream": True,
     "messages": [{"role": "user", "content": f"Summarize in one word. {words}"}]}
ta = threading.Thread(target=stream, args=(a, a_st, a_tx))
ta.start()
while len(a_st) < 20:
    time.sleep(0.01)
tb = threading.Thread(target=stream, args=(b, b_st, []))
tb.start()
tb.join()
ta.join()
gaps = [y - x for x, y in zip(a_st, a_st[1:])]
print(json.dumps({"max_gap_ms": round(max(gaps) * 1000),
                  "sha": hashlib.sha256("".join(a_tx).encode()).hexdigest()[:12]}))
PYEOF

run_arm() {
    local arm="$1" b_words="${2:-1200}" share="${3:-0}"
    local log="$WORK/$arm.log"
    "$BINARY" --serve --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" --prefix-cache-entries 0 \
        --prefill-chunk 1024 --prefill-decode-share "$share" --log-level debug --log-file "$log" \
        >"$WORK/$arm.stdout" 2>&1 &
    SERVER_PID=$!
    local ready=0
    srv_wait_health "http://127.0.0.1:$PORT" 240 0.5 && ready=1
    if [ "$ready" -ne 1 ]; then
        echo "[fail] server did not start ($arm arm)"
        exit 1
    fi
    # Two reps; keep the smaller gap (first rep can carry one-time JIT warmup).
    python3 "$WORK/probe.py" "$PORT" "$b_words" > "$WORK/$arm.r1.json" || { echo "[fail] probe ($arm)"; exit 1; }
    python3 "$WORK/probe.py" "$PORT" "$b_words" > "$WORK/$arm.r2.json" || { echo "[fail] probe ($arm)"; exit 1; }
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    sleep 1
}

run_arm on
run_arm long 6000 0
run_arm share 6000 0.5

if ! grep -q '\[interleave\] engaged' "$WORK/on.log"; then
    echo "[fail] never logged [interleave] engaged — a cold prefill stalls the stream for its whole duration"
    exit 1
fi
echo "[pass] a cold prefill interleaves with the concurrent stream"

# Hosted decode as a fraction of the prefill's wall time, summed over both reps.
hosted_frac() {
    python3 - "$1" <<'PY'
import re, sys
p = d = 0
for line in open(sys.argv[1]):
    m = re.search(r"\[interleave\] prefill (\d+) ms, hosted decode (\d+) ms", line)
    if m:
        p += int(m.group(1)); d += int(m.group(2))
print(f"{d / (p + d):.3f}" if p + d else "0")
PY
}
LONG_FRAC=$(hosted_frac "$WORK/long.log")
SHARE_FRAC=$(hosted_frac "$WORK/share.log")
echo "decode share: hosted fraction today=$LONG_FRAC share0.5=$SHARE_FRAC"
if ! python3 -c "import sys; sys.exit(0 if $SHARE_FRAC > 1.5 * $LONG_FRAC and $SHARE_FRAC > 0 else 1)"; then
    echo "[fail] --prefill-decode-share 0.5 did not raise the decode share of B's prefill (today=$LONG_FRAC share=$SHARE_FRAC)"
    exit 1
fi
echo "[pass] --prefill-decode-share gives decoding streams more of a long prefill"
