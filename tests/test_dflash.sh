#!/bin/bash
# DFlash block-drafter integration test — env-gated on a local target +
# assistant pair (Muse-Glimmer-30B + its DFlash assistant today):
#
#   DFLASH_TEST_MODEL=~/claude-tmp/muse-glimmer/Muse-Glimmer-30B-MLX-Serve-8bit \
#   DFLASH_TEST_DRAFTER=~/.mlx-serve/models/meta-models/Muse-Glimmer-30B-assistant \
#       ./tests/test_dflash.sh
#
# Pins the live contract: the sidecar probe classifies the assistant as
# DFlash (boot log), rounds ENGAGE (`[spec-stats] mode=dflash attempts>0` —
# engagement COUNTS, never output shape), greedy dflash-on equals greedy
# dflash-off byte-for-byte over reasoning+content (always-thinking target),
# per-request enable_drafter:false opts out, and tool calls still parse with
# the drafter engaged.

set -euo pipefail

MODEL="${DFLASH_TEST_MODEL:-}"
DRAFTER="${DFLASH_TEST_DRAFTER:-}"
if [ -z "$MODEL" ] || [ -z "$DRAFTER" ]; then
    echo "SKIP: DFLASH_TEST_MODEL / DFLASH_TEST_DRAFTER not set"
    exit 0
fi
for d in "$MODEL" "$DRAFTER"; do
    if [ ! -f "$d/config.json" ]; then
        echo "FAIL: $d/config.json not found"; exit 1
    fi
done

PORT="${DFLASH_TEST_PORT:-11353}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/dflash_test_serve.XXXXXX)

"$BIN" --model "$MODEL" --drafter "$DRAFTER" --serve --host 127.0.0.1 --port "$PORT" --log-level debug > "$LOG" 2>&1 &
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

pass=0; fail=0
ok()   { echo "PASS $1"; pass=$((pass+1)); }
bad()  { echo "FAIL $1"; shift; for line in "$@"; do echo "  $line"; done; fail=$((fail+1)); }

# [1] The probe classified the sidecar as DFlash at boot.
if grep -q "DFlash drafter ready" "$LOG"; then ok "boot: DFlash sidecar detected"; else bad "boot: DFlash sidecar detected" "$(grep -i drafter "$LOG" | head -3)"; fi

# Greedy request helper: returns reasoning_content + content concatenated.
gen() { # prompt, max_tokens, extra_json_fragment
    curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
        \"model\": \"mlx-serve\",
        \"messages\": [{\"role\": \"user\", \"content\": \"$1\"}],
        \"temperature\": 0.0,
        \"max_tokens\": $2
        $3
    }" | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("reasoning_content") or "") + (m.get("content") or ""))'
}

# [2] Echo-ish greedy round WITH dflash (the default when the sidecar loads) —
# long enough to accumulate spec-stats rounds.
LONG=$(gen "List the numbers from 1 to 15, one per line, then repeat the same list once more." 200 "")
if [ -n "$LONG" ]; then ok "dflash-on generation non-empty"; else bad "dflash-on generation non-empty"; fi

# Equivalence arms: SHORT window (the PLD-equivalence first-30-tokens rule —
# spec verify runs GEMMs at a different width than serial decode, and 8-bit
# near-tie argmax flips are a sanctioned divergence class at long range).
# Fresh prompt (different leading text) so neither arm rides the other's
# prefix-cache entry; the OFF arm turns off PLD too — fully serial.
EQ_PROMPT="Explain in one short paragraph why the sky is blue."
ON=$(gen "$EQ_PROMPT" 30 "")
OFF=$(gen "$EQ_PROMPT" 30 ", \"enable_drafter\": false, \"enable_pld\": false")

# [4] Engagement COUNTS: at least one dflash round ran, with accepts.
STATS=$(grep "mode=dflash" "$LOG" | tail -1)
ATTEMPTS=$(echo "$STATS" | sed -n 's/.*attempts=\([0-9]*\).*/\1/p')
if [ -n "$ATTEMPTS" ] && [ "$ATTEMPTS" -gt 0 ]; then
    ok "spec-stats mode=dflash attempts=$ATTEMPTS ($STATS)"
else
    bad "spec-stats mode=dflash attempts>0" "$(grep spec-stats "$LOG" | tail -3)"
fi

# [5] Greedy equivalence over the first-30-tokens window (reasoning+content).
# A dflash round commits whole blocks, so the ON arm can end a few tokens past
# max_tokens — compare the COMMON PREFIX (the divergence signal), and require
# it to be substantial so a short degenerate answer can't vacuously pass.
if python3 - "$ON" "$OFF" <<'PY'
import sys
on, off = sys.argv[1], sys.argv[2]
n = min(len(on), len(off))
assert n >= 80, f"window too short: {n}"
assert on[:n] == off[:n], f"diverged within the window:\n--- on ---\n{on[:n]}\n--- off ---\n{off[:n]}"
PY
then
    ok "greedy dflash-on == dflash-off (byte-equal common prefix)"
else
    bad "greedy dflash-on == dflash-off" "--- on ---" "$(echo "$ON" | head -c 300)" "--- off ---" "$(echo "$OFF" | head -c 300)"
fi

# [6] The opt-out arm really ran serial (no NEW dflash stats line for it:
# requests 1+2 ran dflash → 2 lines; the opt-out added none).
N_STATS=$(grep -c "mode=dflash" "$LOG")
if [ "$N_STATS" -eq 2 ]; then
    ok "enable_drafter:false opted out (2 dflash stats lines for 3 requests)"
else
    bad "enable_drafter:false opted out" "saw $N_STATS mode=dflash lines (expected 2)"
fi

# [7] Tools still parse with the drafter engaged.
TOOLS=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "mlx-serve",
    "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "temperature": 0.0,
    "max_tokens": 400
}')
if echo "$TOOLS" | python3 -c '
import json, sys
r = json.load(sys.stdin)
calls = r["choices"][0]["message"].get("tool_calls") or []
assert calls, "no tool_calls"
args = json.loads(calls[0]["function"]["arguments"])
assert calls[0]["function"]["name"] == "get_weather", calls[0]["function"]["name"]
assert "city" in args, args
' 2>/dev/null; then
    ok "tool call parses with dflash engaged"
else
    bad "tool call parses with dflash engaged" "$(echo "$TOOLS" | head -c 400)"
fi

# [8] Weight-precision levers ENGAGED, read off the boot log — both are
# load-time decisions, so there is no same-boot A/B for them; the numeric
# guard is the hermetic greedy-equivalence test, which runs default-on.
WLINE=$(grep -o "weights=[^ ]*-bit/gs[0-9]*" "$LOG" | head -1)
if [ -n "$WLINE" ]; then
    ok "assistant quantized at load ($WLINE)"
else
    bad "assistant quantized at load" "$(grep '\[dflash\] loaded' "$LOG" | head -1)"
fi
# The draft head defaults OFF: a narrower head buys bytes the round barely
# notices and costs acceptance (see DEFAULT_DRAFT_HEAD_BITS). The build path
# itself is covered hermetically (draft-head geometry + build policy).
if grep -q "draft lm_head: trunk head" "$LOG"; then
    ok "drafts route through the trunk lm_head by default"
else
    bad "drafts route through the trunk lm_head by default" "$(grep -i 'draft.*lm_head' "$LOG" | head -2)"
fi

# [10] The block is capped to what this machine's verify lanes serve. Without
# the NAX m16 tile the checkpoint's own block (16) is a 4x-cost verify.
BLINE=$(grep -o "DFlash drafter ready (block_size=[0-9]*[^)]*" "$LOG" | head -1)
if echo "$BLINE" | grep -q "capped (no wide verify lane"; then
    ok "block capped on a machine with no wide verify lane ($BLINE)"
elif grep -q "available=true" "$LOG" && echo "$BLINE" | grep -q "block_size=16"; then
    ok "wide verify lane present, checkpoint block kept ($BLINE)"
else
    bad "block resolves against the machine's verify lanes" "$BLINE"
fi

# [11] The assistant context rides the prefix cache. A restore forwards no
# trunk layers, so without it a reused prefix drafts blind — acceptance
# collapsed 92.6% -> 66.5% live. Same prompt twice: the second is a hit, and
# BOTH turns must report the same per-draft rate.
CTX_PROMPT="Repeat this sentence exactly, word for word: the keeper trimmed the wick and wrote three lines in the logbook about the wind and the sea state and the ships that had passed by the point before dawn."
gen "$CTX_PROMPT" 120 "" > /dev/null
RATE_COLD=$(grep -o 'per_draft_pct=[0-9.]*' "$LOG" | tail -1)
gen "$CTX_PROMPT" 120 "" > /dev/null
RATE_HIT=$(grep -o 'per_draft_pct=[0-9.]*' "$LOG" | tail -1)
if grep -q "dflash context restored" "$LOG" && [ "$RATE_COLD" = "$RATE_HIT" ]; then
    ok "assistant context restored from the prefix cache (cold==hit $RATE_HIT)"
else
    bad "assistant context restored from the prefix cache" "cold=$RATE_COLD hit=$RATE_HIT restores=$(grep -c 'dflash context restored' "$LOG")"
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
