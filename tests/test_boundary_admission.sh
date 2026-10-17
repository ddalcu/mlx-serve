#!/bin/bash
# Boundary admission: a small request that arrives while a long prefill runs must not wait for
# the whole prefill.
#
# With --prefill-decode-share the long prefill (B) runs in 1024-row chunks and ticks the streams
# that are already decoding between chunks. A request that arrives mid-prefill (C) is not
# decoding yet: before this change it sat in `pending` until B finished. Now the chunk-boundary
# hook admits it, prefills it and decodes it beside B.
#
# Two arms, same binary:
#   share   --prefill-decode-share 0.5  C's first token comes well before B's prefill ends
#   share0  --prefill-decode-share 0    today's schedule: C waits for B (nothing may change)
# Asserts: [1] share arm: C's first-token wait is under half of B's remaining prefill;
# [2] share0 arm: C waits at least 80% of B's remaining prefill; [3] every request finished
# with text; [4] the share arm logs "[boundary] admitted" and the share0 arm does not;
# [5] cancel arm: a client that disconnects while its request waits for a boundary, and one that
# disconnects from the big prefill after a small request was admitted beside it, leave every other
# request finishing with text and the server answering.
#
# Requires a chat model with at least ~2x the prefill chunk of prompt budget; SKIPs without one.
#   BOUNDARY_TEST_MODEL=/path/to/model BINARY=./zig-out/bin/mlx-serve ./tests/test_boundary_admission.sh [port]

set -u

PORT="${1:-11488}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${BOUNDARY_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E2B-it-qat-4bit}"
B_WORDS="${B_WORDS:-4000}"
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
    echo "[skip] no model at $MODEL (set BOUNDARY_TEST_MODEL)"
    exit 0
fi

cat > "$WORK/probe.py" <<'PYEOF'
import json, random, sys, threading, time, urllib.request

PORT = int(sys.argv[1])
B_WORDS = int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "normal"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def stream(payload, out, abort_after=None):
    """Records request start, the time of the first content chunk, the text and the finish reason.
    `abort_after` seconds after the start the client drops the connection."""
    out["start"] = time.monotonic()
    out["text"] = []
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if abort_after is not None:
            threading.Timer(abort_after, r.close).start()
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    out.setdefault("first", time.monotonic())
                    out["text"].append(d)
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
    out["end"] = time.monotonic()


def payload(content, max_tokens):
    return {"model": "mlx-serve", "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "user", "content": content}]}


def survived(out):
    return bool(out.get("finish")) and bool("".join(out["text"]).strip())


def quiet_stream(payload, out, abort_after=None):
    """A stream whose connection the client may drop: the read error is the expected end."""
    try:
        stream(payload, out, abort_after)
    except Exception:
        out["dropped"] = True
        out["end"] = time.monotonic()


def cancel_case(drop):
    """drop = 'c': the small request C disconnects while it waits for or runs at a boundary.
    drop = 'b': the big prefill B disconnects after C was admitted beside it."""
    a, b, c = {}, {}, {}
    ta = threading.Thread(target=stream, args=(payload("Count from 1 to 400, one number per line.", 400), a))
    ta.start()
    while len(a.get("text", [])) < 20:
        time.sleep(0.01)
    salt = random.randint(0, 10**9)
    words = " ".join(f"w{salt}x{i}" for i in range(B_WORDS))
    tb = threading.Thread(target=quiet_stream, args=(payload(f"Summarize in one word. {words}", 4), b, 2.5 if drop == "b" else None))
    tb.start()
    time.sleep(0.8)
    tc = threading.Thread(target=quiet_stream, args=(payload(f"Reply with the single word ok. ({salt})", 4), c, 0.3 if drop == "c" else None))
    tc.start()
    tc.join()
    tb.join()
    ta.join()
    # The server still answers a fresh request.
    d = {}
    stream(payload("Reply with the single word ok.", 4), d)
    expect_ok = [a, d] + ([b] if drop == "c" else [c])
    return all(survived(x) for x in expect_ok)


if MODE == "cancel":
    print(json.dumps({"cancel_c": cancel_case("c"), "cancel_b": cancel_case("b")}))
    sys.exit(0)

a, b, c = {}, {}, {}
ta = threading.Thread(target=stream, args=(payload("Count from 1 to 400, one number per line.", 800), a))
ta.start()
while len(a.get("text", [])) < 20:
    time.sleep(0.01)
salt = random.randint(0, 10**9)
words = " ".join(f"w{salt}x{i}" for i in range(B_WORDS))
tb = threading.Thread(target=stream, args=(payload(f"Summarize in one word. {words}", 4), b))
tb.start()
time.sleep(0.8)  # inside B's first chunk
tc = threading.Thread(target=stream, args=(payload(f"Reply with the single word ok. ({salt})", 4), c))
tc.start()
tc.join()
tb.join()
ta.join()
b_remaining = b["end"] - c["start"]
print(json.dumps({
    "c_wait_ms": round((c.get("first", c["end"]) - c["start"]) * 1000),
    "b_remaining_ms": round(b_remaining * 1000),
    "b_total_ms": round((b["end"] - b["start"]) * 1000),
    "finished": all(x.get("finish") for x in (a, b, c)),
    "texts": all("".join(x["text"]).strip() for x in (a, b, c)),
}))
PYEOF

run_arm() {
    local arm="$1" share="$2"
    local log="$WORK/$arm.log"
    "$BINARY" --serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" --prefix-cache-entries 0 \
        --max-concurrent 8 --prefill-chunk 1024 --prefill-decode-share "$share" \
        --log-level debug --log-file "$log" >"$WORK/$arm.stdout" 2>&1 &
    SERVER_PID=$!
    local ready=0
    for _ in $(seq 1 480); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
        sleep 0.5
    done
    if [ "$ready" -ne 1 ]; then
        echo "[fail] server did not start ($arm arm)"
        exit 1
    fi
    # Three reps; the first can carry one-time JIT warmup, the median is judged.
    for rep in 1 2 3; do
        python3 "$WORK/probe.py" "$PORT" "$B_WORDS" > "$WORK/$arm.r$rep.json" || { echo "[fail] probe ($arm rep $rep)"; exit 1; }
    done
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    sleep 1
}

run_arm share 0.5
run_arm share0 0

# Cancel races on a share server: the case that drops C, and the case that drops B.
"$BINARY" --serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" --prefix-cache-entries 0 \
    --max-concurrent 8 --prefill-chunk 1024 --prefill-decode-share 0.5 \
    --log-level debug --log-file "$WORK/cancel.log" >"$WORK/cancel.stdout" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 480); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
done
python3 "$WORK/probe.py" "$PORT" "$B_WORDS" cancel > "$WORK/cancel.json" || { echo "[fail] cancel probe crashed"; exit 1; }
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "[fail] server stopped answering after the cancel cases"; exit 1; }
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""

# ratio = C's wait over B's remaining prefill at C's arrival, median over reps 2 and 3.
ratio() {
    python3 - "$WORK/$1.r2.json" "$WORK/$1.r3.json" <<'PY'
import json, sys
r = sorted(json.load(open(p))["c_wait_ms"] / max(1, json.load(open(p))["b_remaining_ms"]) for p in sys.argv[1:])
print(f"{r[len(r) // 2]:.3f}")
PY
}
all_ok() {
    python3 - "$WORK/$1".r?.json <<'PY'
import json, sys
ok = all(json.load(open(p))["finished"] and json.load(open(p))["texts"] for p in sys.argv[1:])
sys.exit(0 if ok else 1)
PY
}
SHARE_RATIO=$(ratio share)
SHARE0_RATIO=$(ratio share0)
echo "boundary admission: C wait / B remaining: share=$SHARE_RATIO share0=$SHARE0_RATIO"

all_ok share || { echo "[fail] share arm: a request ended without text or a finish reason"; exit 1; }
all_ok share0 || { echo "[fail] share0 arm: a request ended without text or a finish reason"; exit 1; }
if ! grep -q '\[boundary\] admitted' "$WORK/share.log"; then
    echo "[fail] share arm never logged [boundary] admitted — nothing was admitted at a chunk boundary"
    exit 1
fi
if grep -q '\[boundary\] admitted' "$WORK/share0.log"; then
    echo "[fail] share0 arm logged [boundary] admitted — share 0 must schedule exactly as before"
    exit 1
fi
if ! python3 -c "import sys; sys.exit(0 if $SHARE0_RATIO >= 0.8 else 1)"; then
    echo "[fail] share0 arm: C no longer waits for B's prefill (ratio $SHARE0_RATIO) — the schedule changed"
    exit 1
fi
if ! python3 -c "import sys; sys.exit(0 if $SHARE_RATIO < 0.5 else 1)"; then
    echo "[fail] share arm: C waited $SHARE_RATIO of B's remaining prefill — not admitted at a chunk boundary"
    exit 1
fi
python3 - "$WORK/cancel.json" <<'PY' || { echo "[fail] cancel arm: $(cat "$WORK/cancel.json")"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r["cancel_c"] and r["cancel_b"] else 1)
PY
echo "[pass] a small request arriving mid-prefill is admitted at a chunk boundary; share 0 is unchanged; cancels at either end leave the rest finishing"
