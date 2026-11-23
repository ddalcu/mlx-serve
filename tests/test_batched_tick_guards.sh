#!/bin/bash
# test_batched_tick_guards.sh — a plain row in the batched decode tick keeps the serial step's per-tick guards.
#   [1] a lone request on MLX_SERVE_FORCE_BATCHED=1 (its row runs the batched kernel) returns the same bytes and finish
#       reason as on a normal server (the serial step; after a forced commit, the same thought and calls), and each
#       guard fires as often on both: a repetition loop is cut (`[loop-stop]`), a reasoning budget closes the thought
#       (`[think-bound]`), and `tool_choice: required` forces the call opener (`[tool-choice]`); the last two SKIP
#       where the serial step never fires them (no bare think tag, no forced opener in the dialect);
#   [2] beside a neighbour that decodes to its cap, every request the serial step cut for a loop is cut again.
#
#   BATCHED_GUARDS_MODEL=<dir> ./tests/test_batched_tick_guards.sh [port]   # default Qwen3-0.6B-4bit
source "$(dirname "$0")/_lib_models.sh"
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
cd "$(dirname "$0")/.."

MODEL="${BATCHED_GUARDS_MODEL:-$(find_model mlx-community/Qwen3-0.6B-4bit || echo "$HOME/.mlx-serve/models/mlx-community/Qwen3-0.6B-4bit")}"
PORT="${1:-11521}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }

WORK=$(mktemp -d "$HOME/claude-tmp/batched-guards.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$WORK/home"
SERVER_PID=""
trap 'srv_stop "$SERVER_PID"' EXIT
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }

start() { # start <log> [env...]
    local log="$1"; shift
    srv_preclean "$PORT"
    env "$@" HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" --no-pld --no-mtp \
        --max-concurrent 8 --prefix-cache-entries 0 --metrics --log-level info > "$log" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 240 0.5 "$SERVER_PID" || { echo "FAIL: server did not start"; tail -20 "$log"; exit 1; }
}
stop() { srv_stop "$SERVER_PID"; SERVER_PID=""; }
post() { curl -s -m "${2:-600}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }

# req <prompt> <max_tokens> <json fragment: sampling and features>
req() { printf '{"model":"m","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,%s}' "$1" "$2" "$3"; }
TOOLS='[{"type":"function","function":{"name":"probe_ping","description":"Acknowledge the probe. Call this with any value.","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}}]'
PLAIN='"enable_thinking":false,"temperature":0'
# One line per answer: finish reason and details, completion tokens, content, reasoning, tool calls.
answer() { python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
ch, u = c["choices"][0], c.get("usage", {})
m = ch["message"]
calls = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
print(json.dumps([ch["finish_reason"], (ch.get("finish_details") or {}).get("type"), u.get("completion_tokens"),
                  m.get("content"), m.get("reasoning_content"), calls]))' "$1" 2>/dev/null; }
looped() { [[ "$(answer "$1" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())[1])' 2>/dev/null)" == repetition_loop ]]; }
# A batched row's forced commit forwards its pending token in the span's chunk (the serial step forwarded it alone), so
# a hybrid trunk's answer after it can round differently: compare what precedes the commit, the finish and the calls.
forced() { python3 -c '
import json, sys
ch = json.load(open(sys.argv[1]))["choices"][0]; m = ch["message"]
print(json.dumps([ch["finish_reason"], m.get("reasoning_content"), bool((m.get("content") or "").strip() or m.get("tool_calls")),
                  [t["function"]["name"] for t in m.get("tool_calls") or []]]))' "$1" 2>/dev/null; }

# Repetition loops (each one falls into it on at least one small model), then the budget and the forced call.
LOOPS=(
    'Write the word go two hundred times.'
    'Continue this text exactly, without stopping: go go go go go go go go go go go go go go go go go go go go go go go go'
    'Print the number 7 three hundred times, separated by spaces.'
)
REQS=()
for p in "${LOOPS[@]}"; do REQS+=("$(req "$p" 300 "$PLAIN")"); done
REQS+=("$(req 'Explain in two sentences why the sky is blue. Think it through carefully first.' 600 '"enable_thinking":true,"temperature":0,"reasoning_budget_tokens":24')")
REQS+=("$(req 'Reply with the single word OK. Do not call any tools.' 256 "$PLAIN,\"tools\":$TOOLS,\"tool_choice\":\"required\"")")

echo "=== [1] a lone plain row: batched kernel == serial step, guards included ($(basename "$MODEL")) ==="
for arm in serial batched; do
    if [[ $arm == batched ]]; then start "$WORK/id-$arm.log" MLX_SERVE_FORCE_BATCHED=1; else start "$WORK/id-$arm.log"; fi
    for i in "${!REQS[@]}"; do post "${REQS[$i]}" > "$WORK/id-$arm-$i.json"; done
    stop
done
grep -q "batched decode engaged" "$WORK/id-batched.log" && ok "forced-batched arm ran the batched kernel" || bad "forced-batched arm ran the batched kernel" "no [batched] ... decode engaged line"
BUDGET_LINE="\[think-bound\] reasoning budget"; FORCED_LINE="\[tool-choice\] call opener forced"
COMMITTED=("$(grep -c "$BUDGET_LINE" "$WORK/id-serial.log")" "$(grep -c "$FORCED_LINE" "$WORK/id-serial.log")")  # after the loops
for i in "${!REQS[@]}"; do
    if [[ $i -ge ${#LOOPS[@]} && ${COMMITTED[$((i - ${#LOOPS[@]}))]} -gt 0 ]]; then
        a=$(forced "$WORK/id-serial-$i.json"); b=$(forced "$WORK/id-batched-$i.json"); what="identical up to the forced commit"
    else
        a=$(answer "$WORK/id-serial-$i.json"); b=$(answer "$WORK/id-batched-$i.json"); what="byte-identical"
    fi
    [[ -n "$a" && "$a" == "$b" ]] && ok "request $i $what (serial vs batched)" || bad "request $i $what" "serial=${a:0:160} batched=${b:0:160}"
done
guard() { # guard <name> <log pattern> [skip-when-serial-is-0]
    local s b
    s=$(grep -c "$2" "$WORK/id-serial.log"); b=$(grep -c "$2" "$WORK/id-batched.log")
    if [[ $s -eq 0 && -n "${3:-}" ]]; then echo "  SKIP $1: the serial step never fired it on this model"; return; fi
    [[ $s -gt 0 && $s -eq $b ]] && ok "$1 fires on both arms ($s)" || bad "$1 fires on both arms" "serial $s, batched $b"
}
guard "the repetition loop cut" "\[loop-stop\]"
guard "the reasoning budget" "$BUDGET_LINE" skip
guard "the forced tool call" "$FORCED_LINE" skip

echo "=== [2] beside a neighbour that decodes to its cap: the same loops are cut ==="
NEIGHBOUR_CAP=600
start "$WORK/group.log"
post "$(req hi 3 "$PLAIN")" >/dev/null
touch "$WORK/sampling"
( peak=0; while [[ -e "$WORK/sampling" ]]; do  # the gauge holds the last group's size; keep its peak
    v=$(curl -s -m 2 "$BASE/metrics" | awk '/^mlx_serve:batched_group_size /{print int($2)}')
    [[ -n "$v" && "$v" -gt "$peak" ]] && peak=$v; sleep 0.2
  done; echo "$peak" > "$WORK/peak" ) & SAMPLER=$!
post "$(req 'Write a long story about the sea.' $NEIGHBOUR_CAP "$PLAIN,\"ignore_eos\":true")" > "$WORK/g-neighbour.json" & CLIENTS=($!)
sleep 0.5
for i in "${!LOOPS[@]}"; do post "${REQS[$i]}" > "$WORK/g-$i.json" & CLIENTS+=($!); done
wait "${CLIENTS[@]}"
rm -f "$WORK/sampling"; wait "$SAMPLER"
stop
peak=$(cat "$WORK/peak" 2>/dev/null)
[[ "${peak:-0}" -ge 2 ]] && ok "a batched group formed (peak size $peak)" || bad "a batched group formed" "batched_group_size peak ${peak:-none}"
n=$(answer "$WORK/g-neighbour.json" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())[2])' 2>/dev/null)
[[ "$n" == "$NEIGHBOUR_CAP" ]] && ok "the neighbour decoded to its cap ($n)" || bad "the neighbour decoded to its cap" "completion_tokens=${n:-none}"
cut=0
for i in "${!LOOPS[@]}"; do
    looped "$WORK/id-serial-$i.json" || continue
    cut=$((cut+1))
    looped "$WORK/g-$i.json" && ok "loop $i cut beside the neighbour" || bad "loop $i cut beside the neighbour" "$(answer "$WORK/g-$i.json" | cut -c1-120)"
done
[[ $cut -gt 0 ]] || bad "a loop the serial step cuts" "none of the loop prompts looped on this model"

echo ""
echo "batched tick guards: $PASS passed, $FAIL failed (logs: $WORK)"
[[ $FAIL -eq 0 ]]
