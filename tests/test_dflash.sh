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
DFLASH_ON_REQS=0
LONG=$(gen "List the numbers from 1 to 15, one per line, then repeat the same list once more." 200 "")
DFLASH_ON_REQS=$((DFLASH_ON_REQS + 1))
if [ -n "$LONG" ]; then ok "dflash-on generation non-empty"; else bad "dflash-on generation non-empty"; fi

# Equivalence arms: SHORT window (the PLD-equivalence first-30-tokens rule —
# spec verify runs GEMMs at a different width than serial decode, and 8-bit
# near-tie argmax flips are a sanctioned divergence class at long range).
# Fresh prompt (different leading text) so neither arm rides the other's
# prefix-cache entry; the OFF arm turns off PLD too — fully serial.
EQ_PROMPT="Explain in one short paragraph why the sky is blue."
ON=$(gen "$EQ_PROMPT" 30 "")
DFLASH_ON_REQS=$((DFLASH_ON_REQS + 1))
OFF=$(gen "$EQ_PROMPT" 30 ", \"enable_drafter\": false, \"enable_pld\": false")

# [4] Engagement COUNTS: at least one dflash round ran, with accepts.
STATS=$(grep "mode=dflash" "$LOG" | tail -1)
ATTEMPTS=$(echo "$STATS" | sed -n 's/.*attempts=\([0-9]*\).*/\1/p')
if [ -n "$ATTEMPTS" ] && [ "$ATTEMPTS" -gt 0 ]; then
    ok "spec-stats mode=dflash attempts=$ATTEMPTS ($STATS)"
else
    bad "spec-stats mode=dflash attempts>0" "$(grep spec-stats "$LOG" | tail -3)"
fi
if GATE_STATE=$(python3 - "$STATS" <<'PY'
import re, sys
line = sys.argv[1]
def field(name):
    match = re.search(rf"\b{name}=([0-9.]+)", line)
    if not match:
        raise SystemExit(f"missing {name}: {line}")
    return float(match.group(1))
avg = field("avg_per_round")
gate = field("gate_min")
disabled = "runtime_disabled=true" in line
if disabled and not avg < gate:
    raise SystemExit(f"disabled despite avg_per_round={avg} >= gate_min={gate}")
print(("disabled" if disabled else "engaged") + f" avg={avg:.2f} gate={gate:.2f}")
PY
); then
    case "$GATE_STATE" in
        disabled*) ok "DFlash gate disabled only below its resolved threshold ($GATE_STATE)" ;;
        engaged*)  ok "DFlash remained engaged for this prose sample ($GATE_STATE)" ;;
        *)         bad "DFlash gate reported a known state" "$GATE_STATE" "$STATS" ;;
    esac
else
    bad "DFlash gate stats obey the disable invariant" "$STATS"
fi

# [5] Greedy equivalence over the exact 30-token window (reasoning+content).
#
# The bar depends on the checkpoint's WIDTH. Spec verify runs the trunk as a
# block-wide `qmm` where serial decode runs `qmv`, and at 4 bits a near-tie
# argmax flips between them — the sanctioned INT4 divergence class. Measured on
# Muse-Glimmer-30B: the 8-bit build is byte-identical 6/6, the 4-bit build
# diverges reproducibly while each arm stays perfectly self-consistent, and the
# divergence is unchanged with MLX_SERVE_SLIDING_BLOCK_TRIM=0.
#
# So: byte-equality at 8-bit or wider; at narrower widths assert what must still
# hold — each arm REPRODUCIBLE (a broken verify or rollback shows as run-to-run
# instability, not as a stable difference) and a long shared prefix — and print
# where they parted. Skipping instead would read as a pass.
QUANT_BITS=$(curl -s -m 10 "$BASE/v1/models" | python3 -c "
import json, re, sys
try:
    d = json.load(sys.stdin)['data'][0]
    m = re.search(r'(\d+)', str((d.get('meta') or {}).get('quantization') or ''))
    print(m.group(1) if m else 8)
except Exception:
    print(8)
" 2>/dev/null || echo 8)

EQ_VERDICT=$(python3 - "$ON" "$OFF" "$QUANT_BITS" <<'PYEQ'
import sys
on, off, bits = sys.argv[1], sys.argv[2], int(sys.argv[3] or 8)
if min(len(on), len(off)) < 80:
    print("fail:window too short"); raise SystemExit
if bits >= 8:
    print("ok:byte-equal" if on == off else "fail:diverged at %d-bit" % bits); raise SystemExit
shared = 0
for a, b in zip(on, off):
    if a != b: break
    shared += 1
print("narrow:%d:%d" % (bits, shared))
PYEQ
)
case "$EQ_VERDICT" in
    ok:*)
        ok "greedy dflash-on == dflash-off (byte-equal exact window, ${QUANT_BITS}-bit)" ;;
    narrow:*)
        SHARED="${EQ_VERDICT##*:}"
        ON2=$(gen "$EQ_PROMPT" 30 "")
        DFLASH_ON_REQS=$((DFLASH_ON_REQS + 1))
        OFF2=$(gen "$EQ_PROMPT" 30 ", \"enable_drafter\": false, \"enable_pld\": false")
        if [ "$ON" != "$ON2" ]; then
            bad "dflash-ON is reproducible at temp 0" "$(echo "$ON" | head -c 200)" "$(echo "$ON2" | head -c 200)"
        elif [ "$OFF" != "$OFF2" ]; then
            bad "serial decode is reproducible at temp 0" "$(echo "$OFF" | head -c 200)" "$(echo "$OFF2" | head -c 200)"
        elif [ "$SHARED" -lt 40 ]; then
            bad "dflash arms agree before the near-tie" "shared only $SHARED chars" "$(echo "$ON" | head -c 200)" "$(echo "$OFF" | head -c 200)"
        else
            ok "greedy dflash arms reproducible + agree for $SHARED chars before a sanctioned ${QUANT_BITS}-bit near-tie"
        fi ;;
    *)
        bad "greedy dflash-on == dflash-off" "$EQ_VERDICT" "--- on ---" "$(echo "$ON" | head -c 300)" "--- off ---" "$(echo "$OFF" | head -c 300)" ;;
esac

# [6] The opt-out arms really ran serial: one `mode=dflash` line per request
# that had dflash ENABLED, and none for the opt-outs. Counted rather than
# hardcoded — the equivalence arm above issues a different number of requests
# depending on the checkpoint's quantization width.
N_STATS=$(grep -c "mode=dflash" "$LOG")
if [ "$N_STATS" -eq "$DFLASH_ON_REQS" ]; then
    ok "enable_drafter:false opted out ($N_STATS dflash stats lines for $DFLASH_ON_REQS dflash-enabled requests)"
else
    bad "enable_drafter:false opted out" "saw $N_STATS mode=dflash lines (expected $DFLASH_ON_REQS)"
fi

# [7] A final accepted block is clipped to the request budget. This checks the
# externally visible accounting; the hermetic Zig test also pins returned ids,
# generated_ids, trunk KV and assistant context to the same exact boundary.
CAP=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "mlx-serve",
    "messages": [{"role": "user", "content": "Write a long numbered list with one short item per line."}],
    "temperature": 0.0,
    "max_tokens": 17
}')
if echo "$CAP" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert r["usage"]["completion_tokens"] == 17, r.get("usage")
assert r["choices"][0]["finish_reason"] == "length", r["choices"][0]
' 2>/dev/null; then
    ok "DFlash output clips exactly at max_tokens=17"
else
    bad "DFlash output clips exactly at max_tokens=17" "$(echo "$CAP" | head -c 500)"
fi

CAP_STREAM=$(curl -s -m 300 -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "mlx-serve",
    "messages": [{"role": "user", "content": "Write a long numbered list with one short item per line."}],
    "temperature": 0.0,
    "max_tokens": 17,
    "stream": true,
    "stream_options": {"include_usage": true}
}')
if python3 - "$CAP" "$CAP_STREAM" <<'PY'
import json, sys
whole = json.loads(sys.argv[1])
msg = whole["choices"][0]["message"]
expected = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
pieces = []
usage = None
for line in sys.argv[2].splitlines():
    if not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        continue
    event = json.loads(data)
    if event.get("usage"):
        usage = event["usage"]
    choices = event.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        pieces.append((delta.get("reasoning_content") or "") + (delta.get("content") or ""))
actual = "".join(pieces)
assert actual == expected, (expected, actual)
assert usage and usage["completion_tokens"] == 17, usage
PY
then
    ok "stream publishes the complete clipped DFlash block"
else
    bad "stream publishes the complete clipped DFlash block" "$(echo "$CAP_STREAM" | tail -c 800)"
fi

# [8] Tools still parse with the drafter engaged.
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

# [9] Weight-precision levers ENGAGED, read off the boot log — both are
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
elif echo "$BLINE" | grep -q "wide_verify_lane=true" && echo "$BLINE" | grep -q "block_size=16"; then
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
