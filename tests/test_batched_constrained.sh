#!/bin/bash
# Batched constrained sampling: grammar, logprobs and penalty requests decode inside the batch,
# and each one still gets its own constraint.
#
# Four requests run together: a JSON-schema request, a logprobs request, a frequency_penalty
# request and a plain one. The plain and penalty rows are short, so the schema and logprobs rows
# outlive them and finish alone: each leaves the batch and continues on its serial arm.
#
# Two arms, same binary:
#   batched  MLX_SERVE_BATCH_CONSTRAINED=1   the default: constrained rows ride the batched forward
#   serial   MLX_SERVE_BATCH_CONSTRAINED=0   each constrained slot decodes alone
# Asserts: [1] batched arm: decode_serial_total for grammar, logprobs and penalty stays 0; the
# serial arm counts grammar and logprobs ticks; [2] both arms: the schema reply parses and has
# every required key; [3] both arms: one logprobs entry per completion token, each with the
# top-1 logprob at temperature 0; [4] every request finished with text.
#
#   BATCHED_CONSTRAINED_TEST_MODEL=/path/to/model BINARY=./zig-out/bin/mlx-serve ./tests/test_batched_constrained.sh [port]

set -u

PORT="${1:-11490}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${BATCHED_CONSTRAINED_TEST_MODEL:-}"
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
    echo "[skip] set BATCHED_CONSTRAINED_TEST_MODEL to a model directory"
    exit 0
fi

cat > "$WORK/probe.py" <<'PYEOF'
import json, sys, threading, urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["tool", "path", "why"],
          "properties": {"tool": {"type": "string", "enum": ["read_file", "grep", "run_tests"]},
                         "path": {"type": "string"}, "why": {"type": "string", "minLength": 120}}}
ASK = "You are a coding agent. Emit ONE tool call as JSON for: find every use of the retry constant, and explain why at length."
cases = {
    "grammar": {"max_tokens": 200, "response_format": {"type": "json_schema", "json_schema": {"name": "call", "strict": True, "schema": SCHEMA}}},
    "logprobs": {"max_tokens": 160, "logprobs": True, "top_logprobs": 1},
    "penalty": {"max_tokens": 16, "frequency_penalty": 0.5},
    "plain": {"max_tokens": 16},
}
res = {}

def ask(kind):
    body = {"model": "m", "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": ASK}], **cases[kind]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        ch = d["choices"][0]
        res[kind] = {"text": ch["message"]["content"] or "", "tokens": d.get("usage", {}).get("completion_tokens", 0),
                     "logprobs": (ch.get("logprobs") or {}).get("content") or []}
    except Exception as e:
        res[kind] = {"text": "", "tokens": 0, "logprobs": [], "error": str(e)}

th = [threading.Thread(target=ask, args=(k,)) for k in cases]
[t.start() for t in th]; [t.join() for t in th]
json.dump(res, open(OUT, "w"))
for k, r in res.items():
    print(f"  {k:8s} tokens={r['tokens']} logprob_entries={len(r['logprobs'])} text={r['text'][:40]!r}" + (f" error={r['error']}" if "error" in r else ""))
PYEOF

serial_counts() {
    curl -s -m 5 "http://127.0.0.1:$PORT/metrics" | python3 -c '
import re, sys
m = {r: 0 for r in ("grammar", "logprobs", "penalty")}
for line in sys.stdin:
    g = re.match(r"\S*decode_serial_total\{reason=\"(grammar|logprobs|penalty)\"\} (\d+)", line)
    if g: m[g.group(1)] = int(g.group(2))
print(" ".join(f"{k}={v}" for k, v in m.items()))'
}

run_arm() {  # $1 = arm name, $2 = MLX_SERVE_BATCH_CONSTRAINED value
    local log="$WORK/$1.log"
    MLX_SERVE_BATCH_CONSTRAINED="$2" "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --max-concurrent 4 --metrics > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 180); do curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 1; done
    echo "[$1] MLX_SERVE_BATCH_CONSTRAINED=$2"
    python3 "$WORK/probe.py" "$PORT" "$WORK/$1.json"
    serial_counts > "$WORK/$1.serial"
    echo "  decode_serial_total: $(cat "$WORK/$1.serial")"
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
}

run_arm batched 1
run_arm serial 0

python3 - "$WORK" <<'PYEOF'
import json, sys
W = sys.argv[1]
arms = {a: json.load(open(f"{W}/{a}.json")) for a in ("batched", "serial")}
ser = {a: dict(kv.split("=") for kv in open(f"{W}/{a}.serial").read().split()) for a in arms}
REQ = ("tool", "path", "why")

def schema_ok(r):
    try:
        o = json.loads(r["text"])
    except Exception:
        return False
    return all(k in o for k in REQ) and o["tool"] in ("read_file", "grep", "run_tests")

def lp_ok(r):
    c = r["logprobs"]
    # Greedy: the chosen token is the top-1, so their logprobs match.
    return len(c) == r["tokens"] > 0 and all(e.get("top_logprobs") and abs(e["top_logprobs"][0]["logprob"] - e["logprob"]) < 1e-5 for e in c)

checks = [
    (f"[1] batched arm serial ticks {ser['batched']} all 0; serial arm {ser['serial']} counts grammar and logprobs",
     all(v == "0" for v in ser["batched"].values()) and int(ser["serial"]["grammar"]) > 0 and int(ser["serial"]["logprobs"]) > 0),
    ("[2] both arms: the schema reply parses with every required key", all(schema_ok(arms[a]["grammar"]) for a in arms)),
    ("[3] both arms: one logprobs entry per token, each the top-1 logprob", all(lp_ok(arms[a]["logprobs"]) for a in arms)),
    ("[4] every request finished with text", all(r["text"].strip() and "error" not in r for a in arms for r in arms[a].values())),
]
for name, ok in checks: print(("[pass] " if ok else "[FAIL] ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
PYEOF
