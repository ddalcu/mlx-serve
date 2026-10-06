#!/usr/bin/env bash
# Boot one Flash Next (qwen4_exp) pack, run a fixed workload (short MTP decode,
# a ~6k-token prefill, two then four concurrent streams), then print which tuned paths
# engaged. Diagnostic helper for per-pack kernel coverage — NOT a test.
#
#   tests/qwen4_engagement.sh <model-dir> [extra mlx-serve flags...]
set -uo pipefail

MODEL="${1:?usage: qwen4_engagement.sh <model-dir> [flags...]}"
shift
PORT="${PORT:-8099}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"
LOG="${LOG:-/tmp/qwen4-engagement-$PORT.log}"
URL="http://127.0.0.1:$PORT"

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
"$BIN" --model "$MODEL" --serve --port "$PORT" --log-level info "$@" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

for _ in $(seq 1 900); do
  curl -sf "$URL/health" >/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "server died; see $LOG"; exit 1; }
  sleep 1
done

chat() { # <max_tokens> <content> [enable_thinking]
  python3 -c 'import json,sys; d={"model":"x","max_tokens":int(sys.argv[1]),"temperature":0,"messages":[{"role":"user","content":sys.argv[2]}]}
if len(sys.argv) > 3: d["enable_thinking"] = sys.argv[3] == "true"
print(json.dumps(d))' "$@" |
    curl -sf "$URL/v1/chat/completions" -H 'content-type: application/json' -d @- >/dev/null
}

chat 128 "Write a Python function that merges two sorted lists, with a docstring."
LONG=$(python3 -c 'print(" ".join(f"Item {i}: the quick brown fox jumps over the lazy dog number {i}." for i in range(450)) + " Summarize the list in one sentence.")')
chat 32 "$LONG"
chat 96 "Explain how a hash map handles collisions." &
A=$!
chat 96 "List five prime numbers and why each is prime." &
wait $A $!
# Four answer-only streams: the MTP group planner drafts in a group, so the head's row kernel runs.
P=()
for q in "the Roman Republic" "the printing press" "the water cycle" "the steam engine"; do
  chat 200 "Write 150 words on $q." false & P+=($!)
done
wait "${P[@]}"

CHECKS=(
  "[hc-prefill] engaged"
  "[qwen4] fused hyper-connection read engaged"
  "[qwen4] two-launch hyper-connection read engaged"
  "[mtp-verify] prepared hyper-connection verifier graphs engaged"
  "[batched] joined hyper-connection verify reads engaged"
  "[batched] GDN verify projections engaged"
  "[batched] attention verify projections engaged"
  "[batched] shared-expert verify projections engaged"
  "[batched] shared-expert gate tail engaged"
  "[mtp] row-axis projection engaged"
  "[moe] fused router kernel engaged: mode=softmax_gate"
  "[moe] affine-4 decode kernels engaged"
  "[vqmm] NAX verify lane engaged"
  "[vqmm] plain-SIMD verify lane engaged"
  "[gdn] packed prework engaged"
  "[qsa-dec] engaged"
  "[qwen4] ngram table warm: started"
  "[qwen4] ple gather:"
)
for c in "${CHECKS[@]}"; do
  line=$(grep -F -m1 -- "$c" "$LOG")
  if [ -n "$line" ]; then printf '  yes  %s\n' "$line" | cut -c1-200; else printf '  NO   %s\n' "$c"; fi
done
grep -F "[mtp] row-axis projection engaged" "$LOG" | sed 's/^/  width: /' | cut -c1-200
grep -E "\[spec-stats\]" "$LOG" | tail -2 | cut -c1-200
echo "log: $LOG"
