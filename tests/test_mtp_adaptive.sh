#!/bin/bash
# test_mtp_adaptive.sh — the adaptive serial switch: past some context a speculative round costs more per token than
# a plain step, so the server measures a plain token per KV bucket and stops speculating where the round loses. Two
# boots of one checkpoint over a ~40k-token prompt generated in-script; MLX_SERVE_ROUND_COST_PERSIST=0.
#
#   MTP_ADAPTIVE_MODEL=<long-context MTP pack> ./tests/test_mtp_adaptive.sh [port]   # SKIPs without one
#
# Pins the invariant, never the model's choice: the switch line or a measured serial cell with the switch on,
# speculation still engaging, no `-> serial` on a short-request burst, and an empty round-cost dir after a --no-mtp
# boot.

set -u
PORT="${1:-11316}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
LOG_A=/tmp/mtp_adaptive_on.log
WORK=$(mktemp -d /tmp/mtp_adaptive.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# First existing candidate wins; MTP_ADAPTIVE_MODEL overrides.
MODEL="${MTP_ADAPTIVE_MODEL:-}"
if [ -z "$MODEL" ]; then
    for cand in \
        "$HOME/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit" \
        "$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit" \
        "$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"; do
        [ -d "$cand" ] && { MODEL="$cand"; break; }
    done
fi
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
    echo "SKIP: no long-context MTP checkpoint found (set MTP_ADAPTIVE_MODEL)"
    exit 0
fi
if [ ! -x "$BIN" ]; then
    echo "SKIP: no server binary at $BIN (build first)"
    exit 0
fi

# ~40k tokens. The context has to clear the top KV bucket's floor (32k) for
# the decision to be made where the feature exists at all.
PROMPT_TOKENS_MIN="${PROMPT_TOKENS_MIN:-30000}"
LONG_CHARS="${LONG_CHARS:-165000}"
# Long enough that the long cells run tens of speculative rounds: the price
# window is 16 rounds and the serial cell needs MIN_SAMPLES ticks.
MAX_TOKENS=200
# Below this the long request cannot have exercised the mechanism, whatever the
# mechanism did — a fixture failure, reported as one.
MIN_LONG_COMPLETION="${MIN_LONG_COMPLETION:-60}"
CTX_SIZE="${CTX_SIZE:-131072}"

PASS=0
FAIL=0
ok()   { echo "PASS [$1]"; PASS=$((PASS+1)); }
bad()  { echo "FAIL [$1]: $2"; FAIL=$((FAIL+1)); }

# Deterministic prompt bodies: an LCG walk over a fixed word list.
python3 - "$WORK" "$LONG_CHARS" "$MAX_TOKENS" <<'PY'
import json, sys, pathlib
work, chars, max_tokens = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

WORDS = ("harbour lantern quiet ledger orchard sandstone drifting compass meadow "
         "tinker rope kettle furrow beacon willow granite shallow ember thistle "
         "marsh cobble weather almanac hollow river bramble shutter pasture flint "
         "lichen tallow bridge cartwright saddle furnace ripple heather quarry "
         "spindle brine oaken hedgerow smithy trellis warren coppice fallow gale").split()

state = 20260904          # fixed seed: the prompt must be a constant
def nxt(n):
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (state >> 33) % n

out, para = [], 0
while sum(len(p) for p in out) < chars:
    para += 1
    sentences = []
    for _ in range(4 + nxt(4)):
        w = [WORDS[nxt(len(WORDS))] for _ in range(8 + nxt(11))]
        w[0] = w[0].capitalize()
        sentences.append(" ".join(w) + ".")
    out.append(f"Section {para}. " + " ".join(sentences) + "\n\n")
long_text = "".join(out)[:chars]

def body(content, mtp):
    return {"model": "default", "stream": False, "temperature": 0, "top_p": 1.0,
            "max_tokens": max_tokens, "enable_thinking": False,
            "enable_pld": False, "enable_drafter": False,
            "enable_mtp": mtp,
            "messages": [{"role": "user", "content": content}]}

# The answer length is part of the fixture: a two-token reply measures nothing.
long_q = (long_text + "\n\nDescribe the material above in detail: its overall "
          "structure, how the sections are numbered, the kind of vocabulary it "
          "uses, and how one section differs from another. Keep writing until "
          "you have covered all four points thoroughly.")
short_q = ("Write a Python function that reverses a linked list in place. "
           "Return only the code.")

(work / "long_serial.json").write_text(json.dumps(body(long_q, False)))
(work / "long_mtp.json").write_text(json.dumps(body(long_q, True)))
(work / "short_mtp.json").write_text(json.dumps(body(short_q, True)))
print(f"generated long prompt: {len(long_q)} chars", file=sys.stderr)
PY
[ -f "$WORK/long_mtp.json" ] || { echo "FAIL: could not generate prompts"; exit 1; }

# Boot C's isolated HOME: the only place this script lets the server persist a round-cost table.
ISO_HOME="$WORK/home"
RC_DIR="$ISO_HOME/.mlx-serve/round-cost"

start_server() { # $1 = log path
    local log="$1"
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    for _ in $(seq 1 30); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 1
    done
    : > "$log"
    # Never read or write the user's round-cost table; the boots must stay independent.
    MLX_SERVE_ROUND_COST_PERSIST=0 MLX_SERVE_DIAG=mtp-trace \
    "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --mtp --kv-quant 8 --ctx-size "$CTX_SIZE" --no-pld \
        --prefix-cache-entries 2 --log-level info >"$log" 2>&1 &
    SERVER_PID=$!
    local model_mb ready_secs
    model_mb=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
    ready_secs=$(( 600 + ${model_mb:-0} / 50 ))
    for _ in $(seq 1 $((ready_secs / 3)) ); do
        grep -q "Model ready (loaded on inference thread)" "$log" && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 3
    done
    echo "FAIL: server did not become ready"; tail -20 "$log"; exit 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 1
    done
}

# req <body-file> <log> <slice-out> -> echoes "<prompt_tokens> <completion_tokens>" and leaves
# this request's own slice of the server log in <slice-out>. Every assertion reads a slice.
req() {
    local bodyf="$1" log="$2" slicef="$3"
    local start end counts
    start=$(wc -c < "$log" | tr -d ' ')
    counts=$(curl -s -m 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary "@$bodyf" |
        python3 -c "
import json, sys
d = json.load(sys.stdin)
if d.get('error'):
    print('ERR', d['error'], file=sys.stderr); raise SystemExit(2)
u = d.get('usage') or {}
c = (d.get('choices') or [{}])[0].get('message', {}).get('content') or ''
print(u.get('prompt_tokens', 0), u.get('completion_tokens', 0), len(c))
") || return 1
    end=$(wc -c < "$log" | tr -d ' ')
    tail -c "+$((start + 1))" "$log" | head -c "$((end - start))" > "$slicef"
    echo "$counts"
}

# The largest measured serial cell reported anywhere in a log (0 if none).
max_serial_cell() { # $1 = log
    grep -o "serial_cell=[0-9.]*" "$1" | cut -d= -f2 |
        sort -g | tail -1 | sed 's/^$/0/'
}
gt_zero() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) > 0 else 1)" "$1"; }

# ── Boot A: the switch ON (default; the env var is simply not set) ────────
echo "== boot A: adaptive serial ON (model $MODEL) =="
start_server "$LOG_A"

# The serial request first: a bucket with no measured plain token cannot be decided.
A_SERIAL=$(req "$WORK/long_serial.json" "$LOG_A" "$WORK/a_serial.slice") ||
    { echo "FAIL: boot A long serial request failed"; stop_server; exit 1; }
A_PROMPT_N=$(echo "$A_SERIAL" | awk '{print $1}')
echo "  long serial: prompt_tokens=$A_PROMPT_N completion=$(echo "$A_SERIAL" | awk '{print $2}')"

if [ "${A_PROMPT_N:-0}" -ge "$PROMPT_TOKENS_MIN" ]; then
    ok "prompt reaches the long-context regime ($A_PROMPT_N >= $PROMPT_TOKENS_MIN tokens)"
else
    bad "prompt length" "only $A_PROMPT_N prompt tokens (< $PROMPT_TOKENS_MIN); raise LONG_CHARS"
fi

A_LONG=$(req "$WORK/long_mtp.json" "$LOG_A" "$WORK/a_long.slice") ||
    { echo "FAIL: boot A long MTP request failed"; stop_server; exit 1; }
A_LONG_N=$(echo "$A_LONG" | awk '{print $2}')
echo "  long mtp:    prompt_tokens=$(echo "$A_LONG" | awk '{print $1}') completion=$A_LONG_N"
[ "$(echo "$A_LONG" | awk '{print $3}')" -gt 0 ] ||
    bad "content" "boot A long MTP request returned empty content"
# A short reply cannot exercise a 16-round window or fill a serial cell, so
# without this the next assertion blames the mechanism for the prompt.
if [ "${A_LONG_N:-0}" -lt "$MIN_LONG_COMPLETION" ]; then
    bad "fixture" "long reply was only $A_LONG_N tokens (< $MIN_LONG_COMPLETION): too few rounds to measure anything — fix the PROMPT, not the controller"
else
    ok "long reply is long enough to price ($A_LONG_N tokens)"
fi

A_SHORT=$(req "$WORK/short_mtp.json" "$LOG_A" "$WORK/a_short.slice") ||
    { echo "FAIL: boot A short request failed"; stop_server; exit 1; }
echo "  short mtp:   prompt_tokens=$(echo "$A_SHORT" | awk '{print $1}') completion=$(echo "$A_SHORT" | awk '{print $2}')"

# llmprobe-shaped burst: the `<2k` bucket only matures over many short requests.
SHORT_BURST="${SHORT_BURST:-12}"
echo "  short burst: $SHORT_BURST requests (llmprobe-shaped)"
burst_start=$(wc -c < "$LOG_A" | tr -d ' ')
bi=1
while [ "$bi" -le "$SHORT_BURST" ]; do
    req "$WORK/short_mtp.json" "$LOG_A" "$WORK/a_burst_$bi.slice" >/dev/null ||
        { echo "FAIL: boot A short burst request $bi failed"; stop_server; exit 1; }
    bi=$((bi + 1))
done
burst_end=$(wc -c < "$LOG_A" | tr -d ' ')
tail -c "+$((burst_start + 1))" "$LOG_A" | head -c "$((burst_end - burst_start))" > "$WORK/a_burst.slice"

stop_server

# [2] Engagement: the long request ran speculative rounds. Without this, [1]
#     could be satisfied by a server that never speculated at all.
if grep -q "\[spec-stats\] mode=mtp" "$WORK/a_long.slice"; then
    ATTEMPTS=$(grep -o "mode=mtp attempts=[0-9]*" "$WORK/a_long.slice" | tail -1 | grep -o "[0-9]*$")
    if [ "${ATTEMPTS:-0}" -gt 0 ]; then
        ok "speculation engaged on the long request (attempts=$ATTEMPTS)"
    else
        bad "engagement" "mode=mtp logged with attempts=0 on the long request"
    fi
else
    bad "engagement" "no '[spec-stats] mode=mtp' for the long request (dispatch hole)"
fi

# [1] The mechanism ran: a decision line, or a measured serial cell.
A_CELL=$(max_serial_cell "$LOG_A")
if grep -q "\[mtp\] adaptive:" "$WORK/a_long.slice"; then
    ok "adaptive controller reported a decision on the long request"
    grep "\[mtp\] adaptive:" "$WORK/a_long.slice" | sed 's/^/      /'
elif gt_zero "$A_CELL"; then
    ok "serial cell measured with the switch on (serial_cell=$A_CELL ms/tok)"
else
    bad "mechanism" "no '[mtp] adaptive:' line and serial_cell=$A_CELL — nothing was measured or decided"
fi

# [3] Short context never switches — the single fixture AND the burst.
if grep -q -- "-> serial" "$WORK/a_short.slice"; then
    bad "short context" "the short code request left speculation (false positive)"
    grep -- "\[mtp\] adaptive:" "$WORK/a_short.slice" | sed 's/^/      /'
else
    ok "short code request stayed on speculation"
fi
BURST_SWITCHES=$(grep -c -- "-> serial" "$WORK/a_burst.slice" 2>/dev/null || true)
if [ "${BURST_SWITCHES:-0}" -gt 0 ]; then
    bad "short burst" "$BURST_SWITCHES of $SHORT_BURST short requests left speculation (the <2k regression)"
    grep -- "\[mtp\] adaptive:" "$WORK/a_burst.slice" | head -5 | sed 's/^/      /'
else
    ok "$SHORT_BURST-request short burst: zero switches (kv floor holds)"
fi

# Boot C: a model that never speculates must not write a round-cost table.
echo "== boot C: --no-mtp, persistence ON, isolated HOME =="
LOG_C=/tmp/mtp_adaptive_nomtp.log
mkdir -p "$ISO_HOME"
rm -rf "$RC_DIR"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
for _ in $(seq 1 30); do
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 1
done
: > "$LOG_C"
HOME="$ISO_HOME" MLX_SERVE_DIAG=mtp-trace MLX_SERVE_ROUND_COST_PERSIST=1 \
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --no-mtp --kv-quant 8 --ctx-size "$CTX_SIZE" --no-pld \
    --prefix-cache-entries 2 --log-level info >"$LOG_C" 2>&1 &
SERVER_PID=$!
c_model_mb=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
c_ready=$(( 600 + ${c_model_mb:-0} / 50 ))
c_up=0
for _ in $(seq 1 $((c_ready / 3)) ); do
    grep -q "Model ready (loaded on inference thread)" "$LOG_C" && { c_up=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 3
done
if [ "$c_up" -ne 1 ]; then
    bad "no-mtp boot" "server did not become ready"
else
    req "$WORK/long_mtp.json" "$LOG_C" "$WORK/c_long.slice" >/dev/null ||
        bad "no-mtp boot" "long request failed"
    req "$WORK/long_serial.json" "$LOG_C" "$WORK/c_serial.slice" >/dev/null ||
        bad "no-mtp boot" "enable_mtp:false request failed"
    stop_server
    RC_FILES=$(ls -1 "$RC_DIR" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${RC_FILES:-0}" -eq 0 ]; then
        ok "--no-mtp boot wrote no round-cost table ($RC_DIR empty)"
    else
        bad "no-mtp boot" "$RC_FILES round-cost file(s) written by a model that never speculates"
        ls -la "$RC_DIR" | sed 's/^/      /'
    fi
    # And it certainly must not have decided anything.
    if grep -q "\[mtp\] adaptive:" "$LOG_C"; then
        bad "no-mtp boot" "an adaptive line appeared with --no-mtp"
    else
        ok "--no-mtp boot made no adaptive decision"
    fi
fi

echo
echo "── $PASS passed, $FAIL failed ── (logs: $LOG_A, $LOG_C)"
[ "$FAIL" -eq 0 ] || exit 1
