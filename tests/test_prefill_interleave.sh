#!/bin/bash
# Prefill-side interleaving: a cold prefill must not stall concurrent decode
# streams for its whole duration.
#
# Pre-interleave, scheduler step 2 ran each admitted prefill to completion
# before any decode tick, so a stream's worst-case inter-token gap equaled the
# ENTIRE incoming prefill (measured: 0.8s at a 9k-token prompt, 7.6s at 53k).
# With chunk-boundary yields the floor is one chunk-forward (~0.15s at
# --prefill-chunk 1024 on E2B). Greedy output must be byte-identical either
# way — interleaving reorders ticks, never math.
#
# Two arms, same binary: MLX_SERVE_PREFILL_INTERLEAVE=0 vs default-on.
# Asserts: [1] the on arm logs "[interleave] engaged" and the off arm doesn't;
# [2] the on arm's max inter-token gap is under half the off arm's;
# [3] stream A's output hash matches across arms.
#
# Requires a small chat model; SKIPs without one.
#   INTERLEAVE_TEST_MODEL=/path/to/model ./tests/test_prefill_interleave.sh [port]

set -u

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
words = " ".join(f"w{i}" for i in range(1200))
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
    local arm="$1" env_val="$2"
    local log="$WORK/$arm.log"
    MLX_SERVE_PREFILL_INTERLEAVE="$env_val" "$BINARY" --serve --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" --prefix-cache-entries 0 \
        --prefill-chunk 1024 --log-level debug --log-file "$log" \
        >"$WORK/$arm.stdout" 2>&1 &
    SERVER_PID=$!
    local ready=0
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
        sleep 0.5
    done
    if [ "$ready" -ne 1 ]; then
        echo "[fail] server did not start ($arm arm)"
        exit 1
    fi
    # Two reps; keep the smaller gap (first rep can carry one-time JIT warmup).
    python3 "$WORK/probe.py" "$PORT" > "$WORK/$arm.r1.json" || { echo "[fail] probe ($arm)"; exit 1; }
    python3 "$WORK/probe.py" "$PORT" > "$WORK/$arm.r2.json" || { echo "[fail] probe ($arm)"; exit 1; }
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    sleep 1
}

run_arm off 0
run_arm on 1

OFF_GAP=$(python3 -c "import json; print(min(json.load(open('$WORK/off.r1.json'))['max_gap_ms'], json.load(open('$WORK/off.r2.json'))['max_gap_ms']))")
ON_GAP=$(python3 -c "import json; print(min(json.load(open('$WORK/on.r1.json'))['max_gap_ms'], json.load(open('$WORK/on.r2.json'))['max_gap_ms']))")
OFF_SHA=$(python3 -c "import json; print(json.load(open('$WORK/off.r1.json'))['sha'])")
ON_SHA=$(python3 -c "import json; print(json.load(open('$WORK/on.r1.json'))['sha'])")

echo "prefill interleave: off_gap=${OFF_GAP}ms on_gap=${ON_GAP}ms off_sha=$OFF_SHA on_sha=$ON_SHA"

if ! grep -q '\[interleave\] engaged' "$WORK/on.log"; then
    echo "[fail] on arm never logged [interleave] engaged — the flag is a silent no-op"
    exit 1
fi
if grep -q '\[interleave\] engaged' "$WORK/off.log"; then
    echo "[fail] off arm logged [interleave] engaged — kill switch is a no-op"
    exit 1
fi
if [ "$ON_SHA" != "$OFF_SHA" ]; then
    echo "[fail] greedy output diverged between arms ($OFF_SHA vs $ON_SHA)"
    exit 1
fi
if [ $((ON_GAP * 2)) -ge "$OFF_GAP" ]; then
    echo "[fail] interleaving did not halve the stall (off=${OFF_GAP}ms on=${ON_GAP}ms)"
    exit 1
fi
echo "[pass] prefill interleaving bounds the concurrent-stream stall"
