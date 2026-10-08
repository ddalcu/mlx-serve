#!/bin/bash
# DeepSeek-V4.1-Flash (deepseek_v41: src/deepseek_v41.zig, the EXL3 repack on
# mlx-stream) live end-to-end, env-gated on a pack (loading takes minutes and
# wants the machine otherwise idle; DSV41_TEST_FLAGS adds server flags, e.g.
# --no-mtp where the DSpark stages do not fit):
#
#   DSV41_TEST_MODEL=~/.mlx-serve/models/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
#       ./tests/test_dsv41.sh
#
# Pins: the embedded template against the release encoder (when staged), the
# default thinking arm and the off arm, stream == non-stream bytes, V4.1's
# spaced DSML tool calls (call -> result -> answer), and single-flight
# admission (module-owned decode state). DSV41_TEST_DSPARK=1 checks DSpark (on
# by default): its rounds engaged, and a second, serial boot (--no-mtp) keeps
# the same greedy bytes (the EXL3 repack on mlx-stream drafts with typical
# acceptance, so there the serial boot only has to answer).
# Hermetic counterparts: the DSV41_TINY fixture tests, `chat: the embedded V4.1
# template ...`, and the format-corpus "dsv41-dsml" family.

set -euo pipefail

MODEL="${DSV41_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then
    echo "SKIP: DSV41_TEST_MODEL not set"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "FAIL: $MODEL/config.json not found"
    exit 1
fi

PORT="${DSV41_TEST_PORT:-11353}"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/dsv41_test_serve.XXXXXX)
EXTRA_FLAGS=()
[ -n "${DSV41_TEST_FLAGS:-}" ] && read -ra EXTRA_FLAGS <<< "$DSV41_TEST_FLAGS"
[ "${DSV41_TEST_SKIP_PREFLIGHT:-0}" = "1" ] && EXTRA_FLAGS+=(--skip-mem-preflight)

ENCODING_DIR="${DSV41_ENCODING_DIR:-$MODEL/encoding}"
if [ -f "$ENCODING_DIR/encoding.py" ]; then
    echo "[0] template A/B vs the release encoder"
    python3 "$(dirname "$0")/dsv41_template_ab.py" --encoding "$ENCODING_DIR" | tail -1
else
    echo "[0] SKIP template A/B (no encoding.py at $ENCODING_DIR)"
fi

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }; }
trap cleanup EXIT

boot() { # extra flags...
    : > "$LOG"
    "$BIN" --model "$MODEL" --serve --port "$PORT" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    echo "waiting for server (load takes minutes)..."
    for _ in $(seq 1 400); do
        curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1
        fi
        sleep 3
    done
    curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "FAIL: no health"; exit 1; }
}

pass=0; fail=0
check() { # name, got, expected-substring
    if grep -qF "$3" <<< "$2"; then
        echo "PASS: $1"; pass=$((pass+1))
    else
        echo "FAIL: $1"; echo "  got:      $2"; echo "  expected: $3"; fail=$((fail+1))
    fi
}
refuse() { # name, got, forbidden-substring
    if grep -qF "$3" <<< "$2"; then
        echo "FAIL: $1 (leaked '$3')"; echo "  got: $2"; fail=$((fail+1))
    else
        echo "PASS: $1"; pass=$((pass+1))
    fi
}
post() { # path, body
    curl -s -m 900 "http://127.0.0.1:$PORT$1" -H 'Content-Type: application/json' -d "$2"
}
text_of() { python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"; }

boot

# [1] Raw completion, greedy.
RAW=$(post /v1/completions '{"model":"mlx-serve","prompt":"The capital of France is","max_tokens":8,"temperature":0}' | text_of)
check "raw greedy answer" "$RAW" "Paris"

# [2] Thinking off: content, no think markup.
OFF=$(post /v1/chat/completions '{"model":"mlx-serve","messages":[{"role":"user","content":"What is the capital of Australia? Answer with just the city name."}],"max_tokens":32,"temperature":0,"enable_thinking":false}')
check "thinking-off content" "$OFF" 'Canberra'
refuse "thinking-off no reasoning" "$OFF" '"reasoning_content"'
refuse "thinking-off no think leak" "$OFF" '</think>'

# [3] A silent request thinks (the template's default), reasoning split out.
ON=$(post /v1/chat/completions '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 3*7? Answer with just the number."}],"max_tokens":1024,"temperature":0,"reasoning_effort":"low"}')
check "thinking-on content" "$ON" '21'
check "thinking-on reasoning present" "$ON" '"reasoning_content"'
refuse "thinking-on no think leak" "$ON" '<think>'

# [4] Stream == non-stream bytes (thinking off, greedy).
SBODY='{"model":"mlx-serve","messages":[{"role":"user","content":"Name three primary colors, comma separated."}],"max_tokens":48,"temperature":0,"enable_thinking":false'
NS=$(post /v1/chat/completions "$SBODY}" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
ST=$(curl -s -m 900 -N "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$SBODY,\"stream\":true}" \
    | python3 -c "
import json,sys
out=''
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]': continue
    for c in json.loads(line[6:]).get('choices', []):
        out += (c.get('delta') or {}).get('content') or ''
print(out)")
if [ "$NS" = "$ST" ] && [ -n "$NS" ]; then
    echo "PASS: stream == non-stream"; pass=$((pass+1))
else
    echo "FAIL: stream != non-stream"; echo "  non-stream: $NS"; echo "  stream:     $ST"; fail=$((fail+1))
fi
refuse "stream no DSML leak" "$ST" 'DSML'

# [5] DSML tool call (V4.1 spells its tags `<｜DSML｜ invoke`).
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Get the current time in a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]'
TC=$(post /v1/chat/completions "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now? Use the tool.\"}],\"tools\":$TOOLS,\"max_tokens\":1024,\"temperature\":0,\"reasoning_effort\":\"low\"}")
check "tool call name" "$TC" '"name":"get_time"'
check "tool call args" "$TC" 'Tokyo'
check "tool finish reason" "$TC" '"finish_reason":"tool_calls"'
refuse "tool call no DSML leak" "$TC" 'DSML'

# [6] Tool round-trip: the result reaches the answer.
RT=$(post /v1/chat/completions "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now? Use the tool.\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{\\\"timezone\\\":\\\"Asia/Tokyo\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"2026-07-31 09:14 JST\"}],\"tools\":$TOOLS,\"max_tokens\":1024,\"temperature\":0,\"reasoning_effort\":\"low\"}")
check "tool round-trip answer" "$RT" '9:14'
refuse "tool round-trip no DSML leak" "$RT" 'DSML'

# [7] Single-flight: a concurrent request queues, never clobbers the module state.
# V4.1's own turn markers: a bare sentence can end at its first token.
SF_BODY='{"model":"mlx-serve","prompt":"<｜begin▁of▁sentence｜><｜User｜>List the first 12 prime numbers, one per line, then explain briefly why 1 is not prime.<｜Assistant｜></think>","max_tokens":128,"temperature":0}'
SOLO=$(post /v1/completions "$SF_BODY" | text_of)
CONC_FILE=$(mktemp /tmp/dsv41_sf_conc.XXXXXX)
post /v1/completions "$SF_BODY" > "$CONC_FILE" &
CONC_PID=$!
sleep 1
MARKER=$(post /v1/chat/completions '{"model":"mlx-serve","messages":[{"role":"user","content":"Reply with exactly: Kangaroo"}],"max_tokens":16,"temperature":0,"enable_thinking":false}')
wait "$CONC_PID" || true
CONC=$(text_of < "$CONC_FILE" 2>/dev/null || echo "<unparseable>")
rm -f "$CONC_FILE"
if [ "$CONC" = "$SOLO" ] && [ -n "$SOLO" ]; then
    echo "PASS: single-flight output byte-equal to solo"; pass=$((pass+1))
else
    echo "FAIL: single-flight output diverged from solo"
    echo "  solo: $(echo "$SOLO" | head -c 300)"; echo "  conc: $(echo "$CONC" | head -c 300)"; fail=$((fail+1))
fi
check "single-flight marker answered" "$MARKER" 'Kangaroo'
refuse "single-flight no cross-request leak" "$CONC" 'Kangaroo'
refuse "no MLX error" "$(cat "$LOG")" '[mlx]'

# [7b] Prefix resume (one conversation): a repeat resumes one token short of its prompt and keeps the
# greedy bytes; a different prompt after it starts fresh, and the first prompt again matches.
R1=$(post /v1/completions "$SF_BODY" | text_of)
R2=$(post /v1/completions "$SF_BODY" | text_of)
check "repeat resumed" "$(grep '\[dsv41\] resumed' "$LOG" | tail -1)" 'resumed'
OTHER=$(post /v1/completions '{"model":"mlx-serve","prompt":"<｜begin▁of▁sentence｜><｜User｜>Name the three primary colors.<｜Assistant｜></think>","max_tokens":32,"temperature":0}')
check "fresh prompt after a resume" "$OTHER" '"finish_reason"'
R3=$(post /v1/completions "$SF_BODY" | text_of)
if [ -n "$R1" ] && [ "$R1" = "$R2" ] && [ "$R1" = "$R3" ]; then
    echo "PASS: resumed and fresh greedy bytes equal"; pass=$((pass+1))
else
    echo "FAIL: resumed or fresh greedy bytes differ"
    for r in "$R1" "$R2" "$R3"; do echo "  $(echo "$r" | head -c 200)"; done; fail=$((fail+1))
fi
# The EXL3 repack runs on mlx-stream.
if [ -f "$MODEL/experts.bin" ]; then
    check "mlx-stream serves the repack" "$(grep '\[mlx-stream\] loaded' "$LOG" || true)" 'tensors from'
fi

# [8] DSpark: the default boot drafted; a serial boot keeps its greedy bytes.
if [ "${DSV41_TEST_DSPARK:-0}" = "1" ]; then
    check "dspark engaged" "$(grep '\[spec-stats\] mode=dspark' "$LOG" || true)" 'mode=dspark attempts='
    # A sampled request drafts too (the host's sampler decides each round), and its seed reproduces it on the same
    # path: both runs cold (a resumed run drafts differently, and speculative sampling spends its draws per draft).
    SAMPLED='{"model":"mlx-serve","prompt":"<｜begin▁of▁sentence｜><｜User｜>Name three rivers in Europe and one fact about each.<｜Assistant｜></think>","max_tokens":96,"temperature":0.8,"top_p":0.95,"seed":7}'
    S1=$(post /v1/completions "$SAMPLED" | text_of)
    post /v1/chat/completions "$SBODY}" > /dev/null
    S2=$(post /v1/completions "$SAMPLED" | text_of)
    check "sampled dspark engaged" "$(grep 'spec=dspark (stochastic' "$LOG" || true)" 'stochastic'
    if [ -n "$S1" ] && [ "$S1" = "$S2" ]; then
        echo "PASS: seeded sampled repeat byte-equal"; pass=$((pass+1))
    else
        echo "FAIL: seeded sampled repeat differs"; echo "  1: $(echo "$S1" | head -c 200)"; echo "  2: $(echo "$S2" | head -c 200)"; fail=$((fail+1))
    fi
    cleanup; SERVER_PID=""
    boot --no-mtp
    SERIAL=$(post /v1/completions "$SF_BODY" | text_of)
    if [ -f "$MODEL/experts.bin" ]; then
        # mlx-stream's lane accepts drafts by typical acceptance: greedy text may differ from serial.
        if [ -n "$SERIAL" ]; then echo "PASS: serial boot answers"; pass=$((pass+1)); else echo "FAIL: serial boot answered nothing"; fail=$((fail+1)); fi
    elif [ "$SERIAL" = "$SOLO" ]; then
        echo "PASS: dspark greedy == serial"; pass=$((pass+1))
    else
        echo "FAIL: dspark greedy != serial"; echo "  dspark: $(echo "$SOLO" | head -c 300)"; echo "  serial: $(echo "$SERIAL" | head -c 300)"; fail=$((fail+1))
    fi
    refuse "serial no MLX error" "$(cat "$LOG")" '[mlx]'
else
    echo "[8] SKIP DSpark arm (DSV41_TEST_DSPARK=1 to run)"
fi

echo
echo "dsv41: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
