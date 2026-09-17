#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECKPOINT=${QWEN4_BF16_STREAM_MODEL:-/Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next}
FIXTURE=${QWEN4_TEACHER_FIXTURE:-/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-60x64}
PORT=${1:-}
LOG=$(mktemp /tmp/mlx-serve-bf16-stream.XXXXXX.log)
BODY1=$(mktemp /tmp/mlx-serve-bf16-stream.XXXXXX.1.json)
BODY2=$(mktemp /tmp/mlx-serve-bf16-stream.XXXXXX.2.json)
BODY3=$(mktemp /tmp/mlx-serve-bf16-stream.XXXXXX.3.json)
PID=

cleanup() {
    if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
    rm -f "$LOG" "$BODY1" "$BODY2" "$BODY3"
}
trap cleanup EXIT

if [[ ! -d "$CHECKPOINT" || ! -f "$FIXTURE/baseline.json" ]]; then
    echo "SKIP bf16 streaming checkpoint or teacher fixture absent"
    exit 0
fi

if [[ -z "$PORT" || ! "$PORT" =~ ^[0-9]+$ || "$PORT" -lt 1 || "$PORT" -gt 65535 ]]; then
    echo "usage: $0 PORT" >&2
    exit 2
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use" >&2
    exit 1
fi

cd "$ROOT"
KLD_JSON=$(mktemp /tmp/mlx-serve-bf16-stream.XXXXXX.kld.json)
./zig-out/bin/mlx-serve kld compare --model "$CHECKPOINT" --fixture "$FIXTURE" --limit 3 --ssd-budget-gb 60 --no-mtp --kv-quant off --json "$KLD_JSON"
jq -e '.mean_kld_to_eos < 0.02 and .mean_top1_to_eos > 0.95' "$KLD_JSON" >/dev/null
rm -f "$KLD_JSON"

./zig-out/bin/mlx-serve --model "$CHECKPOINT" --serve --host 127.0.0.1 --port "$PORT" --expert-cache-gb 60 --no-mtp --kv-quant 8 --ctx-size 65536 --metrics --no-vision >"$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 1200); do
    if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
    if ! kill -0 "$PID" 2>/dev/null; then cat "$LOG"; exit 1; fi
    sleep 1
done
curl --connect-timeout 2 --max-time 5 -fsS "http://127.0.0.1:$PORT/health" >/dev/null
MODEL_ID=$(curl --connect-timeout 2 --max-time 10 -fsS "http://127.0.0.1:$PORT/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')

REQ=$(jq -nc --arg model "$MODEL_ID" '{model:$model,messages:[{role:"user",content:"Write one short sentence about Bangkok."}],temperature:0,max_tokens:16,seed:1234,stream:false}')
curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/chat/completions" >"$BODY1"
curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/chat/completions" >"$BODY2"
jq -e '.choices[0].message.content | type == "string"' "$BODY1" >/dev/null
cmp <(jq -r '.choices[0].message.content' "$BODY1") <(jq -r '.choices[0].message.content' "$BODY2")
LONG_PROMPT=$(for _ in $(seq 1 12); do sed -n '1,$p' "$FIXTURE/prompts/00_wikitext2-test-00-robert-boulter/prompt.txt"; done)
jq -n --arg model "$MODEL_ID" --arg content "$LONG_PROMPT" '{model:$model,messages:[{role:"user",content:$content}],temperature:0,max_tokens:4,stream:false}' | curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/v1/chat/completions" >"$BODY3"
jq -e '.choices[0].message.content | type == "string"' "$BODY3" >/dev/null
grep -E '\[expert-stream\]|Prompt processing|tokens/sec' "$LOG" | tail -80
echo "PASS bf16 streaming kld compare, boot, short prompt, 4k prompt, and greedy determinism"
