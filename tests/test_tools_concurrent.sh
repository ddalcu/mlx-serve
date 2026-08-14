#!/bin/bash
# Guard: concurrent agent loops each get THEIR OWN tool calls.
#
# Batched decode and tool buffering are individually tested and have never been
# tested together. With tools present the server buffers tokens for detection
# per slot, and slots entering a batch mid-generation drain lazy pipeline state
# first — so a cross-slot leak here looks like one client receiving another
# client's arguments, which no single-stream test can see.
#
# N concurrent streams, each planting a DIFFERENT unique token in its prompt and
# asking for one tool call carrying it back. Assertions:
#   • every stream completes (no lost request, [DONE] seen)
#   • every stream's call carries ITS OWN token and no other stream's
#   • every stream's arguments are valid JSON, no markup in visible content
#   • the server is alive and functional afterwards
#
# Usage: CONC_TEST_MODEL=<dir> ./tests/test_tools_concurrent.sh [port]
#   CONC_N=4  — number of concurrent agent loops

set -u

MODEL="${CONC_TEST_MODEL:-}"
PORT="${1:-8136}"
N="${CONC_N:-4}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
PASS=0; FAIL=0
ok()  { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
bad() { echo -e "  ${RED}FAIL${NC} $1"; shift; for l in "$@"; do echo "        $l"; done; FAIL=$((FAIL+1)); }

[ -n "$MODEL" ] || { echo "SKIP: CONC_TEST_MODEL not set"; exit 0; }
[ -f "$MODEL/config.json" ] || { echo "SKIP: no config.json at $MODEL"; exit 0; }
[ -x ./zig-out/bin/mlx-serve ] || { echo "FAIL: build first"; exit 1; }

LOG=$(mktemp /tmp/tools_conc.XXXXXX)
OUTDIR=$(mktemp -d /tmp/tools_conc_out.XXXXXX)
pkill -f "bin/mlx-serve" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve --model "$MODEL" --serve --port "$PORT" \
    --max-concurrent 8 --log-level debug > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -rf "$OUTDIR"; rm -f "$LOG"; }
trap cleanup EXIT
for i in $(seq 1 300); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not start"; exit 1; }

# Distinct, non-overlapping tokens so a cross-slot leak is unambiguous.
TOKENS=(ZULU-1001 YANKEE-2002 XRAY-3003 WHISKEY-4004 VICTOR-5005 TANGO-6006 SIERRA-7007 ROMEO-8008)

echo "launching $N concurrent tool-calling streams"
# Collect the stream PIDs. A bare `wait` also waits on the SERVER started with
# `&` above, which never exits — the run hangs until the harness kills it.
STREAM_PIDS=""
for i in $(seq 0 $((N-1))); do
    TOK="${TOKENS[$i]}"
    python3 - "$BASE" "$TOK" "$OUTDIR/$i.json" <<'PY' &
import json, sys, urllib.request
base, tok, out = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "mlx-serve",
        "messages": [{"role": "user", "content":
            "The session code is %s. Call write_file exactly once to save the session "
            "code to notes.txt, putting the code in the content argument." % tok}],
        "tools": [{"type": "function", "function": {
            "name": "write_file", "description": "Write a file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]}}}],
        "max_tokens": 200, "temperature": 0, "stream": True}
chunks, saw_done, calls, content = [], False, {}, []
try:
    r = urllib.request.urlopen(urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=900)
    for line in r:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            saw_done = True
            continue
        d = json.loads(payload)
        for ch in d.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"name": "", "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
except Exception as e:
    json.dump({"error": str(e)}, open(out, "w")); raise SystemExit
json.dump({"token": tok, "done": saw_done, "calls": list(calls.values()),
           "content": "".join(content)}, open(out, "w"))
PY
    STREAM_PIDS="$STREAM_PIDS $!"
done
for pid in $STREAM_PIDS; do wait "$pid" 2>/dev/null; done

ALL_TOKENS="${TOKENS[*]:0:$N}"
for i in $(seq 0 $((N-1))); do
    TOK="${TOKENS[$i]}"
    F="$OUTDIR/$i.json"
    [ -s "$F" ] || { bad "[stream $i $TOK] produced no result file"; continue; }
    V=$(python3 - "$F" "$TOK" "$ALL_TOKENS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
mine, everyone = sys.argv[2], sys.argv[3].split()
if "error" in d:
    print("err|%s" % d["error"]); raise SystemExit
calls = d["calls"]
if not calls:
    print("nocall|content=%r" % d["content"][:80]); raise SystemExit
try:
    args = json.loads(calls[0]["args"]); valid = 1
except Exception:
    args, valid = {}, 0
blob = json.dumps(args)
carried = 1 if mine in blob else 0
foreign = [t for t in everyone if t != mine and t in blob]
leak = any(t in (d["content"] or "") for t in ("<tool_call", "<|tool_call", "<function="))
print("ok|%d|%d|%d|%s|%d" % (int(d["done"]), valid, carried, ",".join(foreign), int(leak)))
PY
)
    case "$V" in
        err\|*)    bad "[stream $i $TOK] transport error" "${V#err|}"; continue ;;
        nocall\|*) bad "[stream $i $TOK] no tool call in the stream" "${V#nocall|}"; continue ;;
    esac
    DONE=$(echo "$V" | cut -d'|' -f2); VALID=$(echo "$V" | cut -d'|' -f3)
    CARRIED=$(echo "$V" | cut -d'|' -f4); FOREIGN=$(echo "$V" | cut -d'|' -f5)
    LEAK=$(echo "$V" | cut -d'|' -f6)
    [ "$DONE" = "1" ]    && ok "[stream $i $TOK] terminated with [DONE]"     || bad "[stream $i $TOK] no [DONE]"
    [ "$VALID" = "1" ]   && ok "[stream $i $TOK] arguments valid JSON"       || bad "[stream $i $TOK] arguments not valid JSON"
    [ "$CARRIED" = "1" ] && ok "[stream $i $TOK] carried its OWN code"       || bad "[stream $i $TOK] lost its own code"
    [ -z "$FOREIGN" ]    && ok "[stream $i $TOK] carried no other stream's code" \
                         || bad "[stream $i $TOK] CROSS-SLOT LEAK" "found other streams' codes: $FOREIGN"
    [ "$LEAK" = "0" ]    && ok "[stream $i $TOK] no markup in visible content" || bad "[stream $i $TOK] markup leaked"
done

# Batched decode has to have actually happened, or this ran N serial requests.
if grep -qE "batch|concurrent" "$LOG"; then
    ok "server logged batched/concurrent decode activity"
else
    echo "  note: no batch line in the log — streams may have been serialized"
fi

R=$(curl -sf -m 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' 2>/dev/null)
[ -n "$R" ] && ok "server still functional after the concurrent round" \
            || bad "server unresponsive after the concurrent round"

echo
echo "tools under concurrency: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
