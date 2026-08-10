#!/bin/bash
# Integration test: the JSON grammar mask is built PER MODEL.
#
# Live 2026-08-11: the token-byte table backing the mask was a process-wide
# singleton, built from whichever model served the first schema-constrained
# request. With two models resident (hot switching / multi-model registry) the
# second one masked its logits with the FIRST one's vocabulary — ids mean
# different bytes in every vocabulary, so the mask allowed tokens whose real
# bytes are off-schema and `json_object` answered with "## Attributes".
#
# The guard is two-sided: both models must return parseable JSON, and the log
# must show a table built for EACH id (a shared table logs exactly one build).

set -u

MODEL_A=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
MODEL_B=${2:-~/.mlx-serve/models/LiquidAI/LFM2.5-2.6B-MLX-mxfp4}
PORT=${3:-8121}
BASE="http://127.0.0.1:$PORT"
LOG=/tmp/mlx-serve-json-multi-model.log
PASS=0
FAIL=0
TOTAL=0

MODEL_A=$(eval echo "$MODEL_A")
MODEL_B=$(eval echo "$MODEL_B")

for d in "$MODEL_A" "$MODEL_B"; do
    if [ ! -d "$d" ]; then
        echo "SKIP: model not found at $d"
        exit 0
    fi
done

if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }

# Ids are the two-level org/name discovery form — derive them from the paths.
ID_A="$(basename "$(dirname "$MODEL_A")")/$(basename "$MODEL_A")"
ID_B="$(basename "$(dirname "$MODEL_B")")/$(basename "$MODEL_B")"

echo "=== JSON mode across two resident models ==="
echo "A: $ID_A"
echo "B: $ID_B"
echo ""

./zig-out/bin/mlx-serve \
    --model "$MODEL_A" --serve --port $PORT --host 127.0.0.1 --log-level info \
    --model-dir "$(dirname "$(dirname "$MODEL_A")")" \
    --model-dir "$(dirname "$(dirname "$MODEL_B")")" \
    >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    if [ "$i" -eq 60 ]; then echo "FAIL: server did not start within 60s"; exit 1; fi
    sleep 1
done

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then
        PASS=$((PASS + 1)); echo "  PASS: $1"
    else
        FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"
    fi
}

SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}'

ask_json() { # $1=model id, $2=response_format json, $3=prompt
    jq -n --arg m "$1" --argjson rf "$2" --arg p "$3" '{
        model:$m,
        messages:[{role:"user",content:$p}],
        response_format:$rf, temperature:0, max_tokens:128
    }' | curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d @-
}

# json_object only constrains JSON-ness, so the prompt has to bound the shape
# or a chatty model runs past max_tokens and the truncation reads as a failure.
OBJ_PROMPT='Return a JSON object with exactly two keys: "name" (a string) and "age" (a number). No other keys.'
SCHEMA_PROMPT='Invent a person.' 

check() { # stdin = response body
    local body content
    body=$(cat)
    content=$(echo "$body" | jq -r '.choices[0].message.content // ""')
    if [ -z "$content" ]; then echo "empty content"; return; fi
    if echo "$content" | jq -e . >/dev/null 2>&1; then echo ok; else echo "not JSON: ${content:0:80}"; fi
}

# Order is load-bearing: A serves first and builds its table, then B must NOT
# inherit it. The third call re-checks A after B's table exists.
SCHEMA_RF="{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"person\",\"schema\":$SCHEMA,\"strict\":true}}"

for spec in "A json_object|$ID_A|{\"type\":\"json_object\"}|$OBJ_PROMPT" \
            "B json_object|$ID_B|{\"type\":\"json_object\"}|$OBJ_PROMPT" \
            "B json_schema|$ID_B|$SCHEMA_RF|$SCHEMA_PROMPT" \
            "A json_schema|$ID_A|$SCHEMA_RF|$SCHEMA_PROMPT"; do
    NAME=${spec%%|*}; REST=${spec#*|}; MODEL=${REST%%|*}; REST=${REST#*|}; RF=${REST%%|*}; PROMPT=${REST#*|}
    OK=$(ask_json "$MODEL" "$RF" "$PROMPT" | check)
    run_test "$NAME" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
done

# Each model that served a constrained request built its OWN table. A shared
# singleton logs exactly one build line, so this is the class assertion.
for pair in "A|$ID_A" "B|$ID_B"; do
    NAME=${pair%%|*}; ID=${pair#*|}
    if grep -qF "[grammar] building token-byte table for $ID" "$LOG"; then
        run_test "$NAME built its own token-byte table" PASS ""
    else
        run_test "$NAME built its own token-byte table" FAIL "no build line for $ID in $LOG"
    fi
done

echo ""
echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
