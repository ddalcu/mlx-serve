#!/bin/bash
# Muse-Glimmer repetition-loop trigger / regression guard — env-gated on a
# local mirror:
#
#   MUSE_TEST_MODEL=~/.mlx-serve/models/ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit \
#       ./tests/test_muse_repetition.sh
#
# Replays the REAL pi first-turn request captured live 2026-08-11
# (tests/fixtures/muse_pi_doom_request.json — pi 0.84.1, system prompt +
# 4 tools + the doom-clone task; the model restated the request in the
# thinking channel until [loop-stop] cut it). A rep "loops" when the response
# carries finish_details.type == "repetition_loop"; the server log's
# [loop-stop] lines are the cross-check. Any looping rep FAILS — today this
# is the trigger (red); after a fix it is the regression guard (green).
#
# Knobs (rewrite ONLY sampling/effort fields on the fixture at send time):
#   MUSE_REP_REPS        reps to send (default 4)
#   MUSE_REP_TEMP        unset = 0.7 (the incident's effective sampling, from
#                        the app server's launch flags); 0 = greedy
#   MUSE_REP_TOP_P       default 0.95 (incident value)
#   MUSE_REP_TOP_K       unset = not sent (incident had top_k 0; upstream
#                        generation_config recommends 64)
#   MUSE_REP_NO_DAQ=1    boot with --no-decode-attn-quant (bf16 cells — the
#                        lossy side copy only exists on DENSE attention)
#   MUSE_REP_EFFORT      minimal|low|high -> reasoning_effort; off ->
#                        enable_thinking:false (committed to=user channel)
#   MUSE_REP_NO_TOOLS=1  drop the tools array
#   MUSE_REP_NO_DRAFTER=1  boot with --no-drafter (default: the mirror's own
#                        drafter/ is auto-probed -> dflash engaged)
#   MUSE_REP_DUMP_DIR    where looping rep bodies land (default ~/claude-tmp/muse-loop)

set -euo pipefail

MODEL="${MUSE_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then
    echo "SKIP: MUSE_TEST_MODEL not set"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "FAIL: $MODEL/config.json not found"
    exit 1
fi

FIXTURE="$(dirname "$0")/fixtures/muse_pi_doom_request.json"
if [ ! -f "$FIXTURE" ]; then
    echo "FAIL: fixture $FIXTURE not found"
    exit 1
fi

PORT="${MUSE_TEST_PORT:-11357}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/muse_rep_serve.XXXXXX)
REPS="${MUSE_REP_REPS:-4}"
DUMP_DIR="${MUSE_REP_DUMP_DIR:-$HOME/claude-tmp/muse-loop}"
mkdir -p "$DUMP_DIR"

EXTRA_ARGS=()
if [ -n "${MUSE_REP_NO_DRAFTER:-}" ]; then
    EXTRA_ARGS+=(--no-drafter)
fi
if [ -n "${MUSE_REP_NO_DAQ:-}" ]; then
    EXTRA_ARGS+=(--no-decode-attn-quant)
fi

"$BIN" --model "$MODEL" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --serve --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for server..."
for _ in $(seq 1 120); do
    curl -s -m 2 "$BASE/health" > /dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1
    fi
    sleep 3
done

# Rewrite the fixture per env knobs. Only stream/sampling/effort/tools change;
# the messages + system prompt stay the captured bytes.
BODY=$(python3 - "$FIXTURE" <<'EOF'
import json, os, sys
d = json.load(open(sys.argv[1]))
d["stream"] = False
d.pop("stream_options", None)
temp = os.environ.get("MUSE_REP_TEMP", "0.7")
d["temperature"] = float(temp)
d["top_p"] = float(os.environ.get("MUSE_REP_TOP_P", "0.95"))
if os.environ.get("MUSE_REP_TOP_K"):
    d["top_k"] = int(os.environ["MUSE_REP_TOP_K"])
effort = os.environ.get("MUSE_REP_EFFORT", "")
if effort == "off":
    d["enable_thinking"] = False
    d.pop("reasoning_effort", None)
elif effort:
    d["enable_thinking"] = True
    d["reasoning_effort"] = effort
if os.environ.get("MUSE_REP_NO_TOOLS"):
    d.pop("tools", None)
print(json.dumps(d))
EOF
)

echo "config: temp=${MUSE_REP_TEMP:-0.7} top_p=${MUSE_REP_TOP_P:-0.95} effort=${MUSE_REP_EFFORT:-capture} tools=$([ -n "${MUSE_REP_NO_TOOLS:-}" ] && echo off || echo on) drafter=$([ -n "${MUSE_REP_NO_DRAFTER:-}" ] && echo off || echo auto) reps=$REPS"

loops=0
for i in $(seq 1 "$REPS"); do
    R=$(curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")
    VERDICT=$(echo "$R" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("ERROR no-json"); sys.exit()
ch = (d.get("choices") or [{}])[0]
fd = ch.get("finish_details") or {}
usage = d.get("usage") or {}
msg = ch.get("message") or {}
looped = fd.get("type") == "repetition_loop"
print("%s finish=%s tokens=%s" % ("LOOP" if looped else "ok", ch.get("finish_reason"), usage.get("completion_tokens")))
text = (msg.get("reasoning_content") or "") + "\n---content---\n" + (msg.get("content") or "")
sys.stderr.write(text)
' 2>"$DUMP_DIR/rep$i.txt")
    TIER=$(grep '\[loop-stop\]' "$LOG" | tail -1 | grep -o 'tier=[a-z_]*' || true)
    case "$VERDICT" in
        LOOP*)
            loops=$((loops+1))
            echo "FAIL rep $i: $VERDICT $TIER (body -> $DUMP_DIR/rep$i.txt)"
            ;;
        "ERROR no-json")
            echo "FAIL rep $i: non-JSON response"; echo "$R" | head -c 300; echo
            loops=$((loops+1))
            ;;
        *)
            echo "PASS rep $i: $VERDICT"
            rm -f "$DUMP_DIR/rep$i.txt"
            ;;
    esac
done

echo
echo "-- engagement evidence (server log) --"
grep -E 'drafter=enabled|drafter disabled|\[spec-stats\]|\[loop-stop\]' "$LOG" | tail -8 || true
CUTS=$(grep -c '\[loop-stop\]' "$LOG" || true)

echo
echo "muse repetition: $loops/$REPS reps looped ($CUTS loop-stop cuts in server log)"
[ "$loops" -eq 0 ]
