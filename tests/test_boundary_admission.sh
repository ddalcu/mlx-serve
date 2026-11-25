#!/bin/bash
# test_boundary_admission.sh — a small request arriving mid-prefill is admitted at a chunk boundary instead of waiting
# for the whole prefill. Two arms of one binary: --prefill-decode-share 0.5 vs 0, each with a decoding stream, a
# ~4k-token prefill B and a 30-token request C sent 0.8 s into B. [1] with a share, C's first-token wait is under half
# of B's remaining prefill; [2] at share 0 it is at least 80% (that schedule must not change); [3] every request ends
# with text; [4] only the share arm logs `[boundary] admitted`; [5] C dropping while it waits, and B dropping after C
# was admitted, leave every other request finishing and the server answering; [6] at the default share a C whose
# tail is past the boundary limit, sent on /v1/chat/completions, /v1/messages and /api/generate, waits under half of a
# longer B's remaining prefill with `X-MLX-Priority: high` and at least 80% without it, and each route logs its
# priority admission with a prompt between the two limits (B and C are sized through /tokenize).
#
#   BOUNDARY_TEST_MODEL=<dir> BINARY=./zig-out/bin/mlx-serve ./tests/test_boundary_admission.sh [port]
#
# Needs ~2x the prefill chunk of prompt budget; SKIPs without a model.
source "$(dirname "$0")/_lib_models.sh"

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11488}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${BOUNDARY_TEST_MODEL:-$(find_model mlx-community/gemma-4-E2B-it-qat-4bit || echo "$HOME/.mlx-serve/models/mlx-community/gemma-4-E2B-it-qat-4bit")}"
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
import json, random, re, sys, threading, time, urllib.request

PORT = int(sys.argv[1])
B_WORDS = int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "normal"
LOG = sys.argv[4] if len(sys.argv) > 4 else ""
BASE = f"http://127.0.0.1:{PORT}"
ROUTES = {"chat": "/v1/chat/completions", "messages": "/v1/messages", "generate": "/api/generate"}
# The priority arm's B is long enough to host C several times over; C is past the 1024-token boundary limit and well
# under the 8192 priority limit, so it is admitted after about three of B's chunks.
P_B_TOKENS = 24 * 1024
C_TOKENS = 3 * 1024
ADMITTED = re.compile(r"\[boundary\] admitted a (\d+)-token priority request")


def parse(route, line, out):
    """One line of a stream into `out`: text, finish reason and prompt tokens, in the route's wire format."""
    text = None
    if route == "generate":
        obj = json.loads(line)
        text = obj.get("response") or obj.get("thinking")
        if obj.get("done"):
            out["finish"] = obj.get("done_reason") or "stop"
            out["prompt_tokens"] = obj.get("prompt_eval_count")
    elif line.startswith("data: ") and line != "data: [DONE]":
        obj = json.loads(line[6:])
        if route == "messages":
            kind = obj.get("type")
            if kind == "message_start":
                out["prompt_tokens"] = obj["message"]["usage"]["input_tokens"]
            elif kind == "content_block_delta":
                text = obj["delta"].get("text") or obj["delta"].get("thinking")
            elif kind == "message_delta" and obj["delta"].get("stop_reason"):
                out["finish"] = obj["delta"]["stop_reason"]
        else:
            if obj.get("usage"):
                out["prompt_tokens"] = obj["usage"].get("prompt_tokens")
            for ch in obj.get("choices", []):
                # A thinking model streams its reply as reasoning first: either field is decoded text.
                delta = ch.get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content") or text
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
    if text:
        out.setdefault("first", time.monotonic())
        out["text"].append(text)


def stream(payload, out, abort_after=None, priority=False, route="chat"):
    """Records request start, the time of the first content chunk, the text and the finish reason.
    `abort_after` seconds after the start the client drops the connection."""
    out["start"] = time.monotonic()
    out["text"] = []
    headers = {"Content-Type": "application/json"}
    if priority:
        headers["X-MLX-Priority"] = "high"
    req = urllib.request.Request(BASE + ROUTES[route], data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        if abort_after is not None:
            threading.Timer(abort_after, r.close).start()
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                parse(route, line, out)
    out["end"] = time.monotonic()


def payload(content, max_tokens):
    return {"model": "mlx-serve", "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "user", "content": content}]}


def route_payload(route, content, max_tokens):
    if route == "messages":
        return {"model": "mlx-serve", "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
                "messages": [{"role": "user", "content": content}]}
    if route == "generate":
        return {"model": "mlx-serve", "prompt": content, "stream": True,
                "options": {"temperature": 0.0, "num_predict": max_tokens}}
    p = payload(content, max_tokens)
    p["stream_options"] = {"include_usage": True}
    return p


def tokens(text):
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.load(r)["tokens"])


def sized(stem, target):
    """Words `<stem><i>` of about `target` tokens in this model's tokenizer."""
    sample = 200
    n = max(1, round(sample * target / tokens(" ".join(f"{stem}{i}" for i in range(sample)))))
    return " ".join(f"{stem}{i}" for i in range(n))


def priority_admissions():
    with open(LOG, errors="replace") as f:
        return [int(n) for n in ADMITTED.findall(f.read())]


def survived(out):
    return bool(out.get("finish")) and bool("".join(out["text"]).strip())


def wait_tokens(out, thread, n=20, limit=300):
    """Until the stream in `thread` has sent `n` text chunks; it ending or stalling first fails the run."""
    deadline = time.monotonic() + limit
    while len(out.get("text", [])) < n:
        if time.monotonic() > deadline or not thread.is_alive():
            sys.exit(f"stream A sent fewer than {n} text chunks")
        time.sleep(0.01)


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
    wait_tokens(a, ta)
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


def priority_case(priority, route):
    """B prefills beside a decoding A; C, sent on `route` 0.8 s into B, has a tail past the boundary limit."""
    a, b, c = {}, {}, {}
    salt = random.randint(0, 10**9)
    words = sized(f"w{salt}x", P_B_TOKENS)
    c_body = route_payload(route, f"Reply with the single word ok. {sized(f'p{salt}y', C_TOKENS)}", 4)
    ta = threading.Thread(target=stream, args=(payload("Count from 1 to 400, one number per line.", 800), a))
    ta.start()
    wait_tokens(a, ta)
    tb = threading.Thread(target=stream, args=(payload(f"Summarize in one word. {words}", 4), b))
    tb.start()
    time.sleep(0.8)
    tc = threading.Thread(target=stream, args=(c_body, c, None, priority, route))
    tc.start()
    tc.join()
    tb.join()
    ta.join()
    ratio = (c.get("first", c["end"]) - c["start"]) / max(1e-3, b["end"] - c["start"])
    return ratio, all(x.get("finish") for x in (a, b, c)), c.get("prompt_tokens")


if MODE == "cancel":
    print(json.dumps({"cancel_c": cancel_case("c"), "cancel_b": cancel_case("b")}))
    sys.exit(0)

if MODE == "priority":
    out = {}
    for route in ROUTES:
        before = len(priority_admissions())
        with_r, with_ok, c_tokens = priority_case(True, route)
        admitted = priority_admissions()[before:]
        without_r, without_ok, _ = priority_case(False, route)
        out[route] = {"with": with_r, "without": without_r, "finished": with_ok and without_ok,
                      "c_tokens": c_tokens, "admitted": admitted}
    print(json.dumps(out))
    sys.exit(0)

a, b, c = {}, {}, {}
ta = threading.Thread(target=stream, args=(payload("Count from 1 to 400, one number per line.", 800), a))
ta.start()
wait_tokens(a, ta)
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
    srv_wait_health "http://127.0.0.1:$PORT" 480 0.5 && ready=1
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
srv_wait_health "http://127.0.0.1:$PORT" 480 0.5 || true
python3 "$WORK/probe.py" "$PORT" "$B_WORDS" cancel > "$WORK/cancel.json" || { echo "[fail] cancel probe crashed"; exit 1; }
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "[fail] server stopped answering after the cancel cases"; exit 1; }
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""

# Priority on a default-share server: C past the boundary limit, with and without the header.
"$BINARY" --serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" --prefix-cache-entries 0 \
    --max-concurrent 8 --prefill-chunk 1024 \
    --log-level debug --log-file "$WORK/priority.log" >"$WORK/priority.stdout" 2>&1 &
SERVER_PID=$!
srv_wait_health "http://127.0.0.1:$PORT" 480 0.5 || { echo "[fail] server did not start (priority arm)"; exit 1; }
for rep in 1 2 3; do
    python3 "$WORK/probe.py" "$PORT" "$B_WORDS" priority "$WORK/priority.log" > "$WORK/priority.r$rep.json" || { echo "[fail] priority probe (rep $rep)"; exit 1; }
done
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
# Per route, median over reps 2 and 3 of each priority arm; each rep's priority C logged its own admission.
python3 - "$WORK/priority.r2.json" "$WORK/priority.r3.json" <<'PY' || { echo "[fail] priority arm: $(cat "$WORK"/priority.r?.json)"; exit 1; }
import json, sys
rs = [json.load(open(p)) for p in sys.argv[1:]]
ok = True
for route in rs[0]:
    cs = [r[route] for r in rs]
    w = sorted(c["with"] for c in cs)[len(cs) // 2]
    wo = sorted(c["without"] for c in cs)[len(cs) // 2]
    print(f"priority {route}: C wait / B remaining: with={w:.3f} without={wo:.3f}; C prompt tokens {[c['c_tokens'] for c in cs]}")
    for c in cs:
        if not (c["c_tokens"] and 1024 < c["c_tokens"] <= 8192):
            print(f"[fail] {route}: fixture, C has {c['c_tokens']} prompt tokens, wants 1024 < N <= 8192")
            ok = False
        elif not any(1024 < n <= 8192 for n in c["admitted"]):
            print(f"[fail] {route}: no priority admission logged for C (logged {c['admitted']})")
            ok = False
    if not (all(c["finished"] for c in cs) and w < 0.5 and wo >= 0.8):
        print(f"[fail] {route}: with the header C must wait under half of B's remaining prefill, without it at least 80%")
        ok = False
sys.exit(0 if ok else 1)
PY
echo "[pass] a small request arriving mid-prefill is admitted at a chunk boundary; share 0 is unchanged; cancels at either end leave the rest finishing; a priority request past the boundary limit is admitted on every route"
