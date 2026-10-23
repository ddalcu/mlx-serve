#!/bin/bash
# Prefix wait: concurrent requests that share a long prompt prefix must not all prefill it cold.
#
# Three requests with the same ~6k-token prefix (fresh per run) and different questions arrive
# together. Without the wait every request prefills the whole prefix beside the others. With
# MLX_SERVE_PREFIX_WAIT=1 the later ones wait for the first one's commit and restore the prefix.
#
# Two arms, same binary:
#   wait    MLX_SERVE_PREFIX_WAIT=1  at least one later request restores >= 4096 prefix tokens
#   nowait  MLX_SERVE_PREFIX_WAIT=0  today's schedule (nothing may change)
# Asserts: [1] wait arm logs "[prefix-wait]" and the nowait arm does not; [2] wait arm: a later
# request reports cached_tokens >= 4096; [3] wait arm prefills fewer cold prompt tokens in total
# than the nowait arm; [4] every request in both arms finished with text.
#
#   PREFIX_WAIT_TEST_MODEL=/path/to/model BINARY=./zig-out/bin/mlx-serve ./tests/test_prefix_wait.sh [port]

set -u

PORT="${1:-11489}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${PREFIX_WAIT_TEST_MODEL:-}"
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
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
    echo "[skip] set PREFIX_WAIT_TEST_MODEL to a model directory"
    exit 0
fi

cat > "$WORK/probe.py" <<'PYEOF'
import json, random, sys, threading, urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
nonce = random.randrange(1 << 30)
rnd = random.Random(nonce)
words = ["order", "item", "invoice", "ledger", "route", "batch"]
prefix = f"# shared tool catalogue {nonce}\n" + "".join(
    f"\ndef tool_{rnd.choice(words)}_{i}(rows, t={rnd.randint(1, 99)}):\n    return [r for r in rows if r.get('k', 0) > t]\n"
    for i in range(220))
questions = ["Which tool has the smallest threshold?", "Name three tools.", "How many tools are there?"]
res = [None] * len(questions)

def ask(i):
    body = {"model": "m", "max_tokens": 8, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prefix + "\n\n" + questions[i]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        u = d.get("usage", {})
        res[i] = {"text": d["choices"][0]["message"]["content"] or "", "prompt": u.get("prompt_tokens", 0),
                  "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
    except Exception as e:
        res[i] = {"text": "", "prompt": 0, "cached": 0, "error": str(e)}

th = [threading.Thread(target=ask, args=(i,)) for i in range(len(questions))]
[t.start() for t in th]; [t.join() for t in th]
json.dump(res, open(OUT, "w"))
for r in res:
    print(f"  prompt={r['prompt']} cached={r['cached']} text={r['text'][:30]!r}" + (f" error={r['error']}" if "error" in r else ""))
PYEOF

run_arm() {  # $1 = arm name, $2 = MLX_SERVE_PREFIX_WAIT value
    local log="$WORK/$1.log"
    MLX_SERVE_PREFIX_WAIT="$2" "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --max-concurrent 4 > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 180); do curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 1; done
    echo "[$1] MLX_SERVE_PREFIX_WAIT=$2"
    python3 "$WORK/probe.py" "$PORT" "$WORK/$1.json"
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    echo "  [prefix-wait] lines: $(grep -c '\[prefix-wait\]' "$log")"
}

run_arm wait 1
run_arm nowait 0

python3 - "$WORK" <<'PYEOF'
import json, sys
W = sys.argv[1]
on, off = (json.load(open(f"{W}/{a}.json")) for a in ("wait", "nowait"))
lines = {a: open(f"{W}/{a}.log").read().count("[prefix-wait]") for a in ("wait", "nowait")}
cold = lambda rs: sum(r["prompt"] - r["cached"] for r in rs)
checks = [
    ("[1] wait arm logs [prefix-wait], nowait arm does not", lines["wait"] > 0 and lines["nowait"] == 0),
    ("[2] wait arm: a later request restores >= 4096 prefix tokens", sum(r["cached"] >= 4096 for r in on) >= 1),
    (f"[3] cold prompt tokens wait {cold(on)} < nowait {cold(off)}", cold(on) < cold(off)),
    ("[4] every request finished with text", all(r["text"].strip() and "error" not in r for r in on + off)),
]
for name, ok in checks: print(("[pass] " if ok else "[FAIL] ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
PYEOF
