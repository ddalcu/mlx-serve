#!/usr/bin/env bash
# Spec-decode release matrix: compares the shipped /Applications binary
# against the branch binary across the full speculative-decoding feature
# set (none / PLD / MTP / drafter), on each model that supports each
# feature. Sister script to `bench_spec.sh` — that one tunes the gate
# threshold; this one quantifies end-to-end release-over-release deltas.
#
# Output: pipe-separated rows on stdout. stderr has progress + per-cell debug.
#
# Prompts exercise the three regimes the spec-decode features hit:
#   - "echo": output must literally repeat content already in the prompt
#     (PLD/drafter wins — n-gram match against prompt tokens)
#   - "code": agent-style code-edit (function in prompt + small mod requested);
#     the realistic agentic-loop workload, drives the strongest wins
#   - "creative": novel content; the prompt-time gate should disable
#     spec-decode here, expect parity vs `--no-pld`
set -o pipefail

BASELINE_BIN="/Applications/MLX Core.app/Contents/MacOS/mlx-serve"
NEW_BIN="./zig-out/bin/mlx-serve"
MODELS_DIR="$HOME/.mlx-serve/models"
PORT=11240
RUNS=3              # First run = warmup; report run 2 + 3 averaged
CURL_TIMEOUT=120

# ── Prompts ──
# ECHO: classic agent / RAG pattern — paragraph in prompt, output repeats it
ECHO_PROMPT='Below is a paragraph. Repeat it back word-for-word, then add the single sentence "End of recitation." on a new line.

PARAGRAPH:
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. Bright vixens jump; dozy fowl quack. Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz.

Now repeat the paragraph above exactly:'

# CODE: agent code-edit pattern — code in prompt, output repeats it with a small mod
CODE_PROMPT='Here is a Python function. Add a single docstring on the line right after `def`, keep every other line exactly the same as shown. Output ONLY the modified function, no commentary.

def transform(items, threshold):
    result = []
    for x in items:
        if x > threshold:
            result.append(x * 2)
        else:
            result.append(x)
    return result'

# CREATIVE: novel content; the prompt-time gate should disable spec-decode here
CREATIVE_PROMPT='Write an original 120-word short story about a lighthouse keeper who discovers a glowing seashell on the beach. Use vivid sensory detail.'

MAX_TOKENS=220

start_server() {
    local bin="$1" model="$2"; shift 2
    "$bin" --model "$model" --serve --port "$PORT" --log-level warn "$@" \
        >/tmp/mlx_bench_server.log 2>&1 &
    SERVER_PID=$!
    for i in {1..120}; do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    echo "  ERROR: server didn't come up" >&2
    tail -20 /tmp/mlx_bench_server.log >&2
    kill -9 "$SERVER_PID" 2>/dev/null
    return 1
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    sleep 1
}

# Returns "tok/s|prompt_tok|completion_tok"
run_request() {
    local prompt="$1"
    local payload
    payload=$(python3 <<PYEOF
import json
body = {
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': $(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")}],
    'max_tokens': ${MAX_TOKENS},
    'temperature': 0.0,
    'stream': False,
}
print(json.dumps(body))
PYEOF
)

    local t0 t1 elapsed_ms response
    t0=$(python3 -c "import time;print(int(time.time()*1000))")
    response=$(curl -sf -m "$CURL_TIMEOUT" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)
    t1=$(python3 -c "import time;print(int(time.time()*1000))")
    elapsed_ms=$((t1 - t0))

    if [ -z "$response" ]; then
        echo "ERR|0|0"
        return
    fi
    python3 -c "
import json,sys
try:
    r = json.loads(sys.argv[1])
    u = r.get('usage', {})
    pt = u.get('prompt_tokens', 0)
    ct = u.get('completion_tokens', 0)
    ms = float(sys.argv[2])
    toks = ct / (ms/1000.0) if ms > 0 and ct > 0 else 0
    print(f'{toks:.1f}|{pt}|{ct}')
except Exception as e:
    print(f'ERR|0|0')
" "$response" "$elapsed_ms"
}

# Average non-warmup runs
bench_cell() {
    local prompt="$1"
    local results=()
    for i in $(seq 1 $RUNS); do
        local r
        r=$(run_request "$prompt")
        echo "    run $i: $r" >&2
        if [ $i -gt 1 ]; then
            results+=("$r")
        fi
    done
    # average tok/s, take prompt_tok / completion_tok from last
    python3 -c "
import sys
results = sys.argv[1:]
toks = []
last = ''
for r in results:
    parts = r.split('|')
    if parts[0] != 'ERR':
        toks.append(float(parts[0]))
    last = r
if not toks:
    print(last)
else:
    avg = sum(toks)/len(toks)
    parts = last.split('|')
    print(f'{avg:.1f}|{parts[1]}|{parts[2]}')
" "${results[@]}"
}

bench_run() {
    local config_label="$1" model_dir="$2" bin="$3"; shift 3

    echo "==> $config_label" >&2
    if ! start_server "$bin" "$model_dir" "$@"; then
        echo "$config_label|echo|FAIL|0|0"
        echo "$config_label|code|FAIL|0|0"
        echo "$config_label|creative|FAIL|0|0"
        return
    fi

    echo "  echo:" >&2
    echo "$config_label|echo|$(bench_cell "$ECHO_PROMPT")"
    echo "  code:" >&2
    echo "$config_label|code|$(bench_cell "$CODE_PROMPT")"
    echo "  creative:" >&2
    echo "$config_label|creative|$(bench_cell "$CREATIVE_PROMPT")"

    stop_server
}

QWEN="$MODELS_DIR/Qwen3.5-4B-MTPLX-Speed"
GEMMA="$MODELS_DIR/gemma-4-e4b-it-4bit"
GEMMA_DRAFTER="$MODELS_DIR/gemma-4-E4B-it-assistant-bf16"
LFM="$MODELS_DIR/LFM2.5-350M-MLX-8bit"

echo "config|prompt|tok_per_s|prompt_toks|completion_toks"

# Qwen3.5-4B-MTPLX-Speed: baseline + new(no-spec) + PLD + MTP
bench_run "Qwen3.5-4B|baseline_v26.5.3" "$QWEN" "$BASELINE_BIN"
bench_run "Qwen3.5-4B|v26.5.4_nospec"   "$QWEN" "$NEW_BIN" --no-pld
bench_run "Qwen3.5-4B|v26.5.4_PLD"      "$QWEN" "$NEW_BIN"
bench_run "Qwen3.5-4B|v26.5.4_MTP"      "$QWEN" "$NEW_BIN" --no-pld --mtp

# Gemma 4 E4B
bench_run "Gemma4-E4B|baseline_v26.5.3" "$GEMMA" "$BASELINE_BIN"
bench_run "Gemma4-E4B|v26.5.4_nospec"   "$GEMMA" "$NEW_BIN" --no-pld
bench_run "Gemma4-E4B|v26.5.4_PLD"      "$GEMMA" "$NEW_BIN"
bench_run "Gemma4-E4B|v26.5.4_drafter"  "$GEMMA" "$NEW_BIN" --no-pld --drafter "$GEMMA_DRAFTER"

# LFM2.5-350M
bench_run "LFM2.5-350M|baseline_v26.5.3" "$LFM" "$BASELINE_BIN"
bench_run "LFM2.5-350M|v26.5.4_nospec"   "$LFM" "$NEW_BIN" --no-pld
bench_run "LFM2.5-350M|v26.5.4_PLD"      "$LFM" "$NEW_BIN"

echo "DONE" >&2
