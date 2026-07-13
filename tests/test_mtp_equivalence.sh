#!/bin/bash
# MTP (Qwen native multi-token-prediction head) correctness + engagement test.
#
# Contract pinned here:
#   1. ENGAGEMENT — with an MTP sidecar present, every request on
#      /v1/chat/completions (stream + non-stream) and /v1/messages
#      (non-stream) runs speculative rounds: the server log shows
#      `[spec-stats] mode=mtp` with attempts > 0 per request. This is the
#      anti-dispatch-hole check: output-equality alone can't see a silent
#      fallback to regular decode (the drafter shipped exactly that bug).
#   1b. ACCEPTANCE FLOOR — at least one request must report
#      avg_per_round >= 0.5 (accepted tokens per attempt). A structurally
#      broken head (e.g. the delta-encoded-norms trap: sidecar built without
#      the +1 fold-in) still "engages" but accepts ~0 per round before the
#      runtime gate silently falls back to regular decode — equivalence and
#      engagement checks both pass in that state. avg_per_round is the
#      depth-independent floor: healthy measures ~0.7 at depth 1 and ~0.75+
#      at the depth-3 default even on creative temp-0 content where the
#      chained per_draft_pct legitimately dilutes to ~25%.
#   2. EQUIVALENCE — at temp=0 the first $PREFIX_CHARS characters match a
#      --no-mtp baseline byte-for-byte. (Full-output equality is NOT
#      required: INT4 weights make batched verify forwards (qmm) reduce in
#      a different order than single-token decode (qmv), so long greedy
#      tails legitimately diverge — same as PLD/drafter, see CLAUDE.md.)
#
# Usage: MTP_TEST_MODEL=<model-dir> ./tests/test_mtp_equivalence.sh [port]
# Default model: ~/hf-staging/Qwen3.6-27B-4bit-MTP-MLX-Serve (any Qwen 3.5/3.6
# dir with an MTP sidecar — mtp/weights.safetensors, mtp.safetensors, or
# model-mtp.safetensors — works).
#
# MoE trunks (35B-A3B) keep MTP default-OFF per request; set MTP_FORCE_ENABLE=1
# to inject "enable_mtp":true into every request body so engagement +
# acceptance-floor checks exercise the MoE-MLP sidecar arm.

set -u
MODEL="${MTP_TEST_MODEL:-$HOME/hf-staging/Qwen3.6-27B-4bit-MTP-MLX-Serve}"
PORT="${1:-11313}"
BIN="./zig-out/bin/mlx-serve"
# ~24 tokens of prefix. Mirrors the PLD/KV-quant first-N thresholds: INT4
# float-reduction near-ties legitimately flip argmax past ~25-30 tokens
# (observed live at char ~116 on warm prefix-cache requests).
PREFIX_CHARS="${PREFIX_CHARS:-100}"
MAX_TOKENS=120
PROMPT="Write a short story about a robot learning to paint."
# Injected into every request body; empty by default (server defaults apply).
OPTIN=""
if [ "${MTP_FORCE_ENABLE:-0}" = "1" ]; then
    OPTIN='"enable_mtp":true,'
fi

if [ ! -d "$MODEL" ] || { [ ! -f "$MODEL/mtp/weights.safetensors" ] && [ ! -f "$MODEL/mtp.safetensors" ] && [ ! -f "$MODEL/model-mtp.safetensors" ]; }; then
    echo "SKIP: model with MTP sidecar not found at $MODEL"
    exit 0
fi

PASS=0
FAIL=0
LOG=/tmp/mtp_equiv_server.log

start_server() { # $1 = extra flags
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    sleep 1
    # shellcheck disable=SC2086
    "$BIN" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level info $1 >"$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "FAIL: server did not become healthy"; cat "$LOG" | tail -20; exit 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}

chat_nonstream() {
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
        $OPTIN\"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":$MAX_TOKENS,
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" |
        python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'], end='')"
}

chat_stream() {
    curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
        $OPTIN\"model\":\"default\",\"stream\":true,\"temperature\":0,\"max_tokens\":$MAX_TOKENS,
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" |
        python3 -c "
import json, sys
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]': continue
    try: d = json.loads(line[6:])
    except Exception: continue
    for c in d.get('choices', []):
        out.append(c.get('delta', {}).get('content') or '')
print(''.join(out), end='')"
}

messages_nonstream() {
    curl -s "http://127.0.0.1:$PORT/v1/messages" -H 'Content-Type: application/json' -d "{
        $OPTIN\"model\":\"default\",\"stream\":false,\"max_tokens\":$MAX_TOKENS,\"temperature\":0,
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" |
        python3 -c "import json,sys; print(''.join(b.get('text','') for b in json.load(sys.stdin)['content']), end='')"
}

check() { # $1 name, $2 expected-prefix-file, $3 actual-file, $4 expected new mtp engagements (log delta)
    local name="$1" expf="$2" actf="$3" want_engage="$4"
    local exp act
    exp=$(head -c "$PREFIX_CHARS" "$expf")
    act=$(head -c "$PREFIX_CHARS" "$actf")
    if [ -z "$act" ]; then
        echo "FAIL [$name]: empty output"; FAIL=$((FAIL+1)); return
    fi
    if [ "$want_engage" = "yes" ]; then
        local stats
        stats=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
        if [ "$stats" -lt "$ENGAGE_BASE" ] || [ "$stats" -eq "$ENGAGE_BASE" ]; then
            echo "FAIL [$name]: no new '[spec-stats] mode=mtp' log line (engagement hole!)"
            FAIL=$((FAIL+1)); return
        fi
        ENGAGE_BASE=$stats
    fi
    if [ "$exp" != "$act" ]; then
        echo "FAIL [$name]: first $PREFIX_CHARS chars differ from no-mtp baseline"
        echo "  expected: $(echo "$exp" | head -c 80)..."
        echo "  actual:   $(echo "$act" | head -c 80)..."
        FAIL=$((FAIL+1)); return
    fi
    echo "PASS [$name]"
    PASS=$((PASS+1))
}

echo "── baseline server (--no-mtp) ──"
start_server "--no-mtp"
chat_nonstream > /tmp/mtp_base_chat.txt
messages_nonstream > /tmp/mtp_base_msg.txt
if grep -q "mode=mtp" "$LOG"; then
    echo "FAIL: --no-mtp server ran MTP rounds"; FAIL=$((FAIL+1))
else
    echo "PASS [no-mtp baseline clean]"; PASS=$((PASS+1))
fi
stop_server

echo "── MTP server (default-on) ──"
start_server ""
if ! grep -q "MTP head ready" "$LOG"; then
    echo "FAIL: server did not auto-load the MTP sidecar"; tail -5 "$LOG"; FAIL=$((FAIL+1))
else
    echo "PASS [mtp auto-load]"; PASS=$((PASS+1))
fi
ENGAGE_BASE=0
chat_nonstream > /tmp/mtp_on_chat.txt
check "chat non-stream" /tmp/mtp_base_chat.txt /tmp/mtp_on_chat.txt yes
chat_stream > /tmp/mtp_on_chat_stream.txt
check "chat stream" /tmp/mtp_base_chat.txt /tmp/mtp_on_chat_stream.txt yes
messages_nonstream > /tmp/mtp_on_msg.txt
check "messages non-stream" /tmp/mtp_base_msg.txt /tmp/mtp_on_msg.txt yes
# Acceptance floor: a broken head engages but accepts ~0 tokens per round.
# avg_per_round is depth-independent (per_draft_pct divides by depth and
# legitimately dilutes on chained creative drafts at the depth-3 default).
BEST_ACCEPT=$(grep -o 'avg_per_round=[0-9.]*' "$LOG" | cut -d= -f2 | sort -n | tail -1)
if python3 -c "import sys; sys.exit(0 if float('${BEST_ACCEPT:-0}') >= 0.5 else 1)"; then
    echo "PASS [acceptance floor] (best avg_per_round=${BEST_ACCEPT})"; PASS=$((PASS+1))
else
    echo "FAIL [acceptance floor]: best avg_per_round=${BEST_ACCEPT:-none} < 0.5 — head is drafting garbage"
    FAIL=$((FAIL+1))
fi
# Per-request opt-out must fall back to regular decode.
ENGAGE_PRE=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":24,\"enable_mtp\":false,
    \"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" >/dev/null
ENGAGE_POST=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
if [ "$ENGAGE_PRE" -eq "$ENGAGE_POST" ]; then
    echo "PASS [enable_mtp:false opt-out]"; PASS=$((PASS+1))
else
    echo "FAIL [enable_mtp:false opt-out]: MTP ran despite per-request disable"; FAIL=$((FAIL+1))
fi

# EV-controller engagement (dispatch-hole lesson: output equality can't see a
# silent fallback). An ECHO workload is the max-confidence case: past the
# ~10-round warmup the chain confidence clears any tau, so chunk-B extension
# must fire (ext_rounds > 0 in [spec-stats]) under the adaptive default.
ECHO_PROMPT="Repeat the following code block back EXACTLY as written, no commentary: def gcd(a, b):\\n    while b:\\n        a, b = b, a % b\\n    return a\\n\\ndef fib(n, memo={}):\\n    if n in memo: return memo[n]\\n    if n < 2: return n\\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\\n    return memo[n]\\n\\ndef reverse_string(s):\\n    out = ''\\n    for ch in s:\\n        out = ch + out\\n    return out"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    $OPTIN\"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":160,
    \"messages\":[{\"role\":\"user\",\"content\":\"$ECHO_PROMPT\"}]}" >/dev/null
EXT=$(grep -o 'ext_rounds=[0-9]*' "$LOG" | tail -1 | cut -d= -f2)
if [ "${EXT:-0}" -gt 0 ]; then
    echo "PASS [EV chunk-B extension engages on echo] (ext_rounds=$EXT)"; PASS=$((PASS+1))
else
    echo "FAIL [EV chunk-B extension]: ext_rounds=${EXT:-none} on a max-confidence echo — extension path never fired"
    FAIL=$((FAIL+1))
fi
stop_server

echo "── fixed-depth server (MLX_SERVE_MTP_ADAPTIVE=0) ──"
# The env kill switch must fully revert: legacy cap 3 (not the adaptive auto
# cap) and zero chunk-B extensions on the same echo workload.
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
MLX_SERVE_MTP_ADAPTIVE=0 "$BIN" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level info >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    $OPTIN\"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":160,
    \"messages\":[{\"role\":\"user\",\"content\":\"$ECHO_PROMPT\"}]}" >/dev/null
FIXED_STATS=$(grep -o '\[spec-stats\] mode=mtp.*' "$LOG" | tail -1)
FIXED_EXT=$(echo "$FIXED_STATS" | grep -o 'ext_rounds=[0-9]*' | cut -d= -f2)
FIXED_DEPTH=$(echo "$FIXED_STATS" | grep -o ' depth=[0-9]*' | grep -o '[0-9]*')
if [ "${FIXED_EXT:-1}" = "0" ] && [ "${FIXED_DEPTH:-0}" = "3" ]; then
    echo "PASS [MLX_SERVE_MTP_ADAPTIVE=0 reverts to fixed depth 3, no extension]"; PASS=$((PASS+1))
else
    echo "FAIL [adaptive kill switch]: depth=${FIXED_DEPTH:-none} ext_rounds=${FIXED_EXT:-none} (want depth=3 ext_rounds=0)"
    FAIL=$((FAIL+1))
fi
stop_server

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
