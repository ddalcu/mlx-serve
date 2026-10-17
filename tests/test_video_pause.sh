#!/usr/bin/env bash
# Pause / resume of a streaming video job (POST /v1/video/pause).
# Usage: LTX_MODEL=<dir> [SERVER_ARGS=--skip-mem-preflight] ./tests/test_video_pause.sh [port]
#  [1] the stream opens with a `job` event naming the id to pause by
#  [2] a pause lands at the next step boundary: `paused`, then no progress, keepalives only
#  [3] bad bodies 400, unknown ids 404
#  [4] resume: `resumed`, progress continues, the clip completes
#  [5] a pause withdrawn at once never holds the run
#  [6] a finished job's id pauses nothing
set -uo pipefail
PORT="${1:-11339}"
MODEL="${LTX_MODEL:-$(ls -d ~/.mlx-serve/models/*/LTX-2.5-MLX-Serve-8bit 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no LTX model (set LTX_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
LOG="$(mktemp -t mlxserve-pause.XXXXXX)"

"$BIN" --model "$MODEL" --serve --port "$PORT" ${SERVER_ARGS:-} >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$LOG"' EXIT
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
  sleep 2
done

python3 - "$PORT" <<'PY'
import http.client, json, sys, time
port = int(sys.argv[1])
BODY = {"prompt": "a red fox running through a snowy forest", "num_frames": 9, "height": 256,
        "width": 384, "steps": 6, "seed": 7, "stream": True}

def post(body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", "/v1/video/pause", body if isinstance(body, str) else json.dumps(body),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, r.read()

def fail(msg):
    print("FAIL:", msg); sys.exit(1)

def stream():
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/video/generations", json.dumps(BODY), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200: fail(f"generation http {r.status}: {r.read()[:300]}")
    while True:
        line = r.fp.readline()
        if not line: return
        line = line.decode().rstrip("\n")
        if line.startswith(": "): yield {"type": "keepalive", "at": time.monotonic()}
        elif line.startswith("data: "):
            ev = json.loads(line[6:]); ev["at"] = time.monotonic(); yield ev

# [1]-[4]
events, job, paused_at, resumed_at = [], None, None, None
for ev in stream():
    events.append(ev)
    t = ev["type"]
    if t == "job":
        job = ev["id"]
        print("PASS [1]: job event first:", job) if len(events) == 1 else fail("job event is not first")
    elif t == "progress" and ev.get("step", 0) >= 1 and paused_at is None and job:
        st, _ = post({"id": job, "pause": True})
        if st != 200: fail(f"pause http {st}")
        paused_at = -1
    elif t == "paused":
        paused_at = ev["at"]
        time.sleep(7)  # held: progress must stop, keepalives keep the socket warm
        for bad, want in [("not json", 400), ({"id": job}, 400), ({"id": "123", "pause": False}, 404)]:
            st, _ = post(bad)
            if st != want: fail(f"{bad!r} -> {st}, want {want}")
        print("PASS [3]: malformed 400, unknown id 404")
        st, _ = post({"id": job, "pause": False})
        if st != 200: fail(f"resume http {st}")
    elif t == "resumed":
        resumed_at = ev["at"]
    elif t == "error":
        fail(ev)
if paused_at in (None, -1): fail("no paused event")
held = [e for e in events if paused_at < e["at"] < (resumed_at or 1e18)]
if any(e["type"] == "progress" for e in held): fail("progress during pause")
if not any(e["type"] == "keepalive" for e in held): fail("no keepalive during a 7 s pause")
print(f"PASS [2]: held {resumed_at - paused_at:.1f} s at a step boundary, keepalives only")
after = [e for e in events if resumed_at and e["at"] > resumed_at]
if not resumed_at or not any(e["type"] == "progress" for e in after) or events[-1]["type"] != "complete":
    fail("did not resume to completion: " + str([e["type"] for e in events[-5:]]))
print("PASS [4]: resumed and completed")

# [5]
events, job = [], None
for ev in stream():
    events.append(ev)
    if ev["type"] == "job":
        job = ev["id"]
    elif ev["type"] == "progress" and ev.get("step", 0) == 1:
        a, _ = post({"id": job, "pause": True}); b, _ = post({"id": job, "pause": False})
        if (a, b) != (200, 200): fail(f"pause/withdraw http {a}/{b}")
pz = [e["at"] for e in events if e["type"] in ("paused", "resumed")]
if events[-1]["type"] != "complete": fail("withdrawn run did not complete")
if len(pz) == 2 and pz[1] - pz[0] > 1.0: fail(f"withdrawn pause held {pz[1]-pz[0]:.1f} s")
print("PASS [5]: a withdrawn pause does not hold the run")

# [6]
st, _ = post({"id": job, "pause": True})
if st != 404: fail(f"finished job id -> {st}, want 404")
print("PASS [6]: a finished job's id pauses nothing")
PY
