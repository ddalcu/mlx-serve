#!/bin/bash
# sweep_all_archs.sh — exercise every supported architecture against mlx-serve
# AND benchmark each against mlx-lm. One model per turn — we never load
# mlx-serve and mlx-lm at the same time (90 GB+ checkpoints OOM the GPU when
# loaded twice in one process, per CLAUDE.md gotcha).
#
# For each model the script:
#   1. Boots mlx-serve on a free port.
#   2. Quick sanity chat (one turn).
#   3. Three-turn agent-style memory test: plant fact, intervene, recall.
#   4. Timed bench prompt (prefill + decode tok/s).
#   5. Tears the server down.
#   6. Loads the same model via mlx-lm Python and times the same decode budget
#      (skipped for GGUF — mlx-lm doesn't load GGUF).
#
# Output: a markdown table written to BENCHMARK_RESULTS.md plus stdout summary.
#
# Override which models are tested via SWEEP_MODELS=<csv of logical names>.
# Skip mlx-lm comparison via SWEEP_SKIP_MLX_LM=1.

set -uo pipefail

cd "$(dirname "$0")/.."

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11295}"
MAX_TOKENS="${MAX_TOKENS:-128}"
BENCH_PROMPT="${BENCH_PROMPT:-You are a careful, thorough technical writer. Write an 80-word vivid description of a thunderstorm over the Pacific Ocean from the perspective of a sailor on a small wooden boat. Include sensory details: the smell of ozone, the sound of thunder rolling across open water, the rhythm of waves, the visual contrast of dark clouds against silver lightning. Avoid clichés. Use specific concrete imagery. The writing should feel immediate and physical. Begin directly with the description; do not preface it with an introduction. End the description naturally without a closing summary.}"
SKIP_MLX_LM="${SWEEP_SKIP_MLX_LM:-0}"
RESULTS="${RESULTS:-BENCHMARK_RESULTS.md}"

# Logical-name | display | path | type (mlx|gguf)
# Each entry is checked for existence; missing ones skip cleanly.
MODELS=(
    "gemma4-e2b-4bit|Gemma 4 E2B (4-bit)|$HOME/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit|mlx"
    "gemma4-e4b-4bit|Gemma 4 E4B (4-bit)|$HOME/.lmstudio/models/mlx-community/gemma-4-e4b-it-4bit|mlx"
    "gemma4-26b-moe-4bit|Gemma 4 26B-A4B MoE (4-bit)|$HOME/.mlx-serve/models/gemma-4-26b-a4b-it-4bit|mlx"
    "gemma4-31b-4bit|Gemma 4 31B (4-bit)|$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit|mlx"
    "qwen36-27b-4bit|Qwen 3.6 27B dense (4-bit)|$HOME/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit|mlx"
    "qwen36-35b-moe-ud|Qwen 3.6 35B-A3B MoE UD (4-bit)|$HOME/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit|mlx"
    "dsv4-flash-gguf|DeepSeek-V4-Flash (GGUF, ds4)|$HOME/projects/agents/ds4/ds4flash.gguf|gguf"
)

# Allow SWEEP_MODELS=gemma4-e2b-4bit,dsv4-flash-gguf to override.
if [[ -n "${SWEEP_MODELS:-}" ]]; then
    IFS=',' read -r -a WANTED <<< "$SWEEP_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do
            if [[ "$name" == "$w" ]]; then FILTERED+=("$entry"); fi
        done
    done
    MODELS=("${FILTERED[@]}")
fi

# Bail if binary missing.
if [[ ! -x "$BINARY" ]]; then
    echo "[fatal] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

# Make sure nothing else is occupying our port.
pkill -f 'mlx-serve --serve' >/dev/null 2>&1 || true
sleep 1

echo "# Architecture sweep — $(date '+%Y-%m-%d %H:%M')" > "$RESULTS"
echo "" >> "$RESULTS"
echo "Binary: \`$BINARY\` ($(du -h "$BINARY" | awk '{print $1}'))" >> "$RESULTS"
echo "Methodology: warm-vs-warm — each engine gets an 8-token warmup on the same prompt before the timed run." >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| Architecture | mlx-serve sanity | 3-turn recall | prefill tok/s | decode tok/s | mlx-lm decode tok/s | Δ |" >> "$RESULTS"
echo "|---|---|---|---|---|---|---|" >> "$RESULTS"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'

wait_for_health() {
    local port="$1" timeout="${2:-60}" pid="$3"
    for ((i=1; i<=timeout; i++)); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        sleep 1
    done
    return 1
}

run_one() {
    local name="$1" display="$2" path="$3" type="$4"

    echo -e "${BLUE}=== $display ===${NC}"

    if [[ "$type" == "mlx" && ! -d "$path" ]]; then
        echo -e "${YELLOW}[skip]${NC} model dir not found: $path"
        echo "| $display | skip (missing) | — | — | — | — | — |" >> "$RESULTS"
        return 0
    fi
    if [[ "$type" == "gguf" && ! -f "$path" ]]; then
        echo -e "${YELLOW}[skip]${NC} GGUF not found: $path"
        echo "| $display | skip (missing) | — | — | — | — | — |" >> "$RESULTS"
        return 0
    fi

    local log
    log=$(mktemp)
    echo "  booting on port $PORT (log: $log)…"
    "$BINARY" --model "$path" --serve --port "$PORT" --max-tokens "$MAX_TOKENS" > "$log" 2>&1 &
    local sp=$!
    # mlx-serve safety: scheduler does eager warmup; large MoE models can take
    # 90+ s on cold weights pages, so allow a generous timeout.
    if ! wait_for_health "$PORT" 180 "$sp"; then
        echo -e "${RED}[fail]${NC} server didn't come up; tail of log:"
        tail -30 "$log" | sed 's/^/    /'
        kill "$sp" 2>/dev/null
        wait "$sp" 2>/dev/null
        echo "| $display | FAIL (boot) | — | — | — | — | — |" >> "$RESULTS"
        return 1
    fi
    echo "  up."

    # ── 1. Sanity ──
    local sanity_resp
    sanity_resp=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 2+2? Reply with one short sentence."}],"max_tokens":32,"temperature":0.0}' \
        2>/dev/null || true)
    local sanity_text
    sanity_text=$(printf '%s' "$sanity_resp" | python3 -c 'import json,sys
try: r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])
except: print("")' 2>/dev/null)
    local sanity_status="—"
    if [[ -n "$sanity_text" ]]; then
        sanity_status="ok"
        echo "  sanity: $sanity_text" | head -c 140; echo
    else
        sanity_status="FAIL"
        echo -e "  ${RED}sanity FAIL${NC}"
    fi

    # ── 2. Three-turn recall (plant → distract → recall) ──
    # We track history client-side and call /v1/chat/completions each turn.
    local recall_status
    recall_status=$(SWEEP_PORT="$PORT" python3 - <<'PY' || true
import json, os, sys, urllib.request

PORT = int(os.environ["SWEEP_PORT"])
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PLANT = "My favorite codename is BLUE-MERIDIAN. Please confirm you noted it."
DISTRACT = "Briefly, what is 17 times 13?"
RECALL = "What's the favorite codename I told you a moment ago? Reply with just the codename."

def call(messages, n=64):
    body = json.dumps({
        "model": "mlx-serve",
        "messages": messages,
        "max_tokens": n,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<ERR {e}>"

history = []
history.append({"role": "user", "content": PLANT})
r1 = call(history, 64)
history.append({"role": "assistant", "content": r1})

history.append({"role": "user", "content": DISTRACT})
r2 = call(history, 32)
history.append({"role": "assistant", "content": r2})

history.append({"role": "user", "content": RECALL})
r3 = call(history, 24)

ok = "BLUE-MERIDIAN" in r3.upper().replace("‐", "-").replace("–", "-")
status = "ok" if ok else "FAIL"
print(status + "|" + r3.replace("|", "/").strip()[:160])
PY
)
    local recall_label
    recall_label="${recall_status%%|*}"
    local recall_text="${recall_status#*|}"
    if [[ "$recall_label" == "ok" ]]; then
        echo "  recall: ok → $recall_text"
    else
        echo -e "  ${RED}recall FAIL${NC} → $recall_text"
    fi

    # ── 3. Timed bench prompt — prefill + decode tok/s from response usage ──
    # The mlx-lm comparison side warms with an 8-token gen on the SAME prompt
    # shape before its timed measurement, so apples-to-apples requires the
    # same here. The sanity + recall turns above don't count as warmup for
    # the bench prompt — they use different inputs and a different chat-history
    # shape, so JIT-compile / page-fault costs for the bench prompt's shape
    # still land on the first measurement otherwise.
    local bench_payload warmup_payload
    bench_payload=$(BENCH_PROMPT="$BENCH_PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import json,os; print(json.dumps({"model":"mlx-serve","messages":[{"role":"user","content":os.environ["BENCH_PROMPT"]}],"max_tokens":int(os.environ["MAX_TOKENS"]),"temperature":0.0}))')
    warmup_payload=$(BENCH_PROMPT="$BENCH_PROMPT" python3 -c 'import json,os; print(json.dumps({"model":"mlx-serve","messages":[{"role":"user","content":os.environ["BENCH_PROMPT"]}],"max_tokens":8,"temperature":0.0}))')
    # Warmup — discard timing.
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$warmup_payload" >/dev/null 2>&1 || true
    # Snapshot log size BEFORE the timed bench so we only read new lines for
    # the measured request.
    local log_before
    log_before=$(wc -c <"$log")
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$bench_payload" >/dev/null 2>&1 || true
    # Wait for the post-request log line to land.
    sleep 0.4
    local bench_log_slice
    bench_log_slice=$(tail -c +$((log_before + 1)) "$log")
    local prefill_tps decode_tps
    prefill_tps=$(printf '%s' "$bench_log_slice" | grep -oE 'prefill: [0-9.]+ tok/s' | tail -1 | awk '{print $2}')
    decode_tps=$(printf '%s' "$bench_log_slice" | grep -oE 'decode: [0-9.]+ tok/s' | tail -1 | awk '{print $2}')
    : "${prefill_tps:=—}"
    : "${decode_tps:=—}"
    echo "  bench: prefill=$prefill_tps tok/s  decode=$decode_tps tok/s"

    # Tear down server before mlx-lm comparison (memory-tight on 96 GB; mlx-lm
    # loads its own copy).
    kill "$sp" 2>/dev/null
    wait "$sp" 2>/dev/null
    sleep 2

    # ── 4. mlx-lm comparison (skip for GGUF — mlx-lm doesn't load .gguf) ──
    local mlx_lm_tps="—" delta="—"
    if [[ "$type" == "mlx" && "$SKIP_MLX_LM" != "1" ]]; then
        echo "  running mlx-lm comparison…"
        local mlx_lm_log
        mlx_lm_log=$(mktemp)
        BENCH_PROMPT="$BENCH_PROMPT" MAX_TOKENS="$MAX_TOKENS" MODEL_PATH="$path" python3 - >"$mlx_lm_log" 2>&1 <<'PY' || true
import os, sys, time, contextlib, io, traceback
prompt = os.environ["BENCH_PROMPT"]
max_tokens = int(os.environ["MAX_TOKENS"])
path = os.environ["MODEL_PATH"]
try:
    # Silence mlx-lm progress output (it pollutes the tok/s parse).
    with contextlib.redirect_stderr(io.StringIO()):
        from mlx_lm import load, generate
        model, tok = load(path)
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        # Warm — also primes Metal JIT for the same shapes.
        _ = generate(model, tok, prompt=prompt_ids, max_tokens=8, verbose=False)
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=prompt_ids, max_tokens=max_tokens, verbose=False)
        dt = time.perf_counter() - t0
        gen_ids = tok.encode(out)
        n = len(gen_ids)
        tps = (n / dt) if (dt > 0 and n > 0) else 0.0
        print(f"TPS={tps:.1f} N={n} DT={dt:.2f}")
except Exception:
    traceback.print_exc(file=sys.stdout)
PY
        mlx_lm_tps=$(grep -oE 'TPS=[0-9.]+' "$mlx_lm_log" | head -1 | sed 's/TPS=//')
        : "${mlx_lm_tps:=—}"
        if [[ "$mlx_lm_tps" == "—" ]]; then
            echo "    (mlx-lm failed; log: $mlx_lm_log)" >&2
            head -5 "$mlx_lm_log" >&2
        else
            rm -f "$mlx_lm_log"
        fi
        if [[ "$decode_tps" != "—" && "$mlx_lm_tps" != "—" ]]; then
            delta=$(python3 -c "print(f'{(float('$decode_tps') / float('$mlx_lm_tps') - 1) * 100:+.1f}%')")
        fi
        echo "  mlx-lm: $mlx_lm_tps tok/s   Δ vs mlx-serve: $delta"
    fi

    echo "| $display | $sanity_status | $recall_label | $prefill_tps | $decode_tps | $mlx_lm_tps | $delta |" >> "$RESULTS"
    rm -f "$log"
    return 0
}

trap 'pkill -f "mlx-serve --serve" 2>/dev/null || true' EXIT

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name display path type <<< "$entry"
    run_one "$name" "$display" "$path" "$type" || true
    pkill -f "mlx-serve --serve" 2>/dev/null || true
    sleep 3
done

echo
echo "=== Sweep complete ==="
echo "Results: $RESULTS"
echo
cat "$RESULTS"
