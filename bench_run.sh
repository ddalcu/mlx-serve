#!/bin/bash
# bench_run.sh — single-cell engine bench.
#
# Brings up ONE engine on ONE model with ONE spec config, sends matched-params
# OpenAI chat-completions for both prefill and decode measurements (and
# optionally extra spec-decode prompts), tears down, prints pipe-separated
# rows on stdout.
#
# Usage:
#   bench_run.sh \
#     --engine <mlx-serve|mlx-lm|lmstudio> \
#     --model <path|key>          \  # mlx-serve/mlx-lm: local path; lmstudio: model-key
#     --spec  <none|pld|drafter> \
#     [--drafter-dir <path>]      \  # required for spec=drafter (mlx-serve only)
#     [--port N]                  \  # default 11240; lmstudio always 1234
#     [--runs N]                  \  # default 3; run 1 is warmup, dropped
#     [--max-tokens N]            \  # default 192
#     [--ctx-size N]              \  # default 4096 (passed to engine + load)
#     [--binary <path>]           \  # mlx-serve binary; default ./zig-out/bin/mlx-serve
#     [--label <text>]            \  # echoed in output rows; default = engine
#     [--prompts <file>]          \  # extra spec-decode prompts, one per line: name=text
#     [--no-prefill]              \  # skip prefill measurement
#     [--no-decode]               \  # skip decode measurement
#     [--quiet]                      # suppress stderr progress
#
# Output (stdout):
#   label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|hardware|notes
#
# `hardware` is auto-detected from the host (chip + RAM) so CSVs from different
# Macs can be merged without losing provenance. Override via --hardware <tag> if
# you know better (e.g. running on a remote rig that misreports sysctl values).
#
# Exit codes: 0 OK, 1 fatal error (engine failed to start, etc).

set -uo pipefail

# ── Defaults ──
ENGINE=""
MODEL=""
SPEC="none"
DRAFTER_DIR=""
PORT=""
RUNS=3
MAX_TOKENS=192
CTX_SIZE=4096
BINARY="./zig-out/bin/mlx-serve"
LABEL=""
PROMPTS_FILE=""
DO_PREFILL=1
DO_DECODE=1
DO_ECHO=0
QUIET=0
HARDWARE=""

# ── Hardware detection ──
# Tags rows so CSVs from different Macs can be merged. Format: "<chip>-<ram>gb",
# e.g. "Apple-M1-Pro-32gb" or "Apple-M4-Max-128gb". Spaces in the chip name are
# replaced with dashes so the tag is one shell-friendly token.
detect_hardware() {
    local chip ram_gb
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    if [[ "$(uname)" == "Darwin" ]]; then
        ram_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))
    else
        ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    fi
    echo "${chip// /-}-${ram_gb}gb"
}

# Fixed prompts so prefill/decode rates are reproducible.
PREFILL_PROMPT="Explain the following topics in extreme detail: $(python3 -c "print(', '.join([f'topic {i} about science and technology and its impact on human civilization throughout history' for i in range(1,50)]))")"
DECODE_PROMPT="Write a detailed essay about quantum computing"
# Heavy-echo: the canonical workload for spec-decode (PLD/drafter). Output
# is the input verbatim, so n-gram lookahead nearly always hits.
ECHO_PROMPT="Repeat the following paragraph back to me word for word, exactly as written, with no additional commentary. Then add the single sentence \"End of recitation.\" on a new line. PARAGRAPH: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. Bright vixens jump; dozy fowl quack. Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz. Now repeat the paragraph above exactly:"

# Common request body knobs we send to every engine. Apples-to-apples:
# greedy decoding, no thinking, fixed ctx, no system prompt.
COMMON_BODY_FIELDS='"temperature":0.0,"top_p":1.0,"stream":false'

# ── Arg parse ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)       ENGINE="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --spec)         SPEC="$2"; shift 2 ;;
        --drafter-dir)  DRAFTER_DIR="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --runs)         RUNS="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --ctx-size)     CTX_SIZE="$2"; shift 2 ;;
        --binary)       BINARY="$2"; shift 2 ;;
        --label)        LABEL="$2"; shift 2 ;;
        --prompts)      PROMPTS_FILE="$2"; shift 2 ;;
        --no-prefill)   DO_PREFILL=0; shift ;;
        --no-decode)    DO_DECODE=0; shift ;;
        --echo)         DO_ECHO=1; shift ;;
        --quiet)        QUIET=1; shift ;;
        --hardware)     HARDWARE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)              echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$ENGINE" ]] && { echo "missing --engine" >&2; exit 1; }
[[ -z "$MODEL"  ]] && { echo "missing --model"  >&2; exit 1; }
[[ -z "$LABEL"  ]] && LABEL="$ENGINE"
[[ -z "$PORT"   ]] && PORT=$([[ "$ENGINE" == "lmstudio" ]] && echo 1234 || echo 11240)
[[ -z "$HARDWARE" ]] && HARDWARE=$(detect_hardware)

dbg() { [[ "$QUIET" == "1" ]] || echo "  $*" >&2; }
fail() { echo "$*" >&2; exit 1; }

# ── Engine lifecycle ──
SERVER_PID=""

start_engine() {
    case "$ENGINE" in
        mlx-serve)
            local extra=""
            case "$SPEC" in
                none)    extra="--no-pld" ;;
                pld)     extra="" ;; # default-on
                drafter) [[ -z "$DRAFTER_DIR" ]] && fail "spec=drafter needs --drafter-dir"
                         extra="--no-pld --drafter $DRAFTER_DIR" ;;
                *)       fail "bad spec: $SPEC" ;;
            esac
            "$BINARY" --model "$MODEL" --serve --port "$PORT" \
                --ctx-size "$CTX_SIZE" --log-level info $extra \
                >/tmp/bench_engine.log 2>&1 &
            SERVER_PID=$!
            ;;
        mlx-lm)
            # mlx_lm.server is OpenAI-compatible; spec-decode via --draft-model
            # only (we don't expose that here — matches "drafter only on mlx-serve"
            # design). Other specs map to plain serve.
            python3 -m mlx_lm.server --model "$MODEL" --port "$PORT" \
                --temp 0 --max-tokens "$MAX_TOKENS" --log-level WARNING \
                >/tmp/bench_engine.log 2>&1 &
            SERVER_PID=$!
            ;;
        lmstudio)
            # Server is global; start it idempotently and load this model.
            lms server start --port "$PORT" >/tmp/bench_engine.log 2>&1 || true
            # Wait for server up before loading
            for i in $(seq 1 60); do
                curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
                sleep 0.5
            done
            # Unload anything currently loaded so the bench cell is clean.
            lms unload --all >>/tmp/bench_engine.log 2>&1 || true
            # Load with explicit ctx-length for apples-to-apples.
            lms load "$MODEL" --context-length "$CTX_SIZE" --gpu max --ttl 86400 \
                >>/tmp/bench_engine.log 2>&1 || fail "lms load failed for '$MODEL'"
            return 0
            ;;
        *) fail "unknown engine: $ENGINE" ;;
    esac

    # Wait for /v1/models or /health to respond (mlx-serve uses /health).
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
        || curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    fail "engine $ENGINE did not start on port $PORT (see /tmp/bench_engine.log)"
}

stop_engine() {
    case "$ENGINE" in
        mlx-serve|mlx-lm)
            [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
            wait "$SERVER_PID" 2>/dev/null
            SERVER_PID=""
            ;;
        lmstudio)
            lms unload --all >/dev/null 2>&1 || true
            # Leave the server running across cells; helper script stops it once.
            ;;
    esac
    sleep 0.5
}

# ── Request body construction ──
# Builds a JSON request body for the given prompt + max_tokens. Adds engine-
# specific spec-decode hints so we test the requested config (e.g., enable_pld).
build_body() {
    local prompt="$1" mt="$2"
    local extra=""
    if [[ "$ENGINE" == "mlx-serve" ]]; then
        # enable_thinking:false applies on Qwen3.5/3.6 dense; ignored elsewhere.
        # enable_*: per-request override of the spec-decode mode the server was started with.
        case "$SPEC" in
            none)    extra=',"enable_pld":false,"enable_drafter":false,"enable_thinking":false' ;;
            pld)     extra=',"enable_pld":true,"enable_drafter":false,"enable_thinking":false' ;;
            drafter) extra=',"enable_pld":false,"enable_drafter":true,"enable_thinking":false' ;;
        esac
    elif [[ "$ENGINE" == "lmstudio" ]]; then
        # LM Studio honours top-level chat-template hints in `chat_template_kwargs`.
        extra=',"chat_template_kwargs":{"enable_thinking":false}'
    fi
    # mlx-lm: no per-request thinking knob; chat template default applies.
    # mlx-lm.server treats `model` as an HF id and will 404 if it doesn't match
    # what was loaded at startup; lmstudio routes by its own model key. Echo
    # the engine-specific identifier we used at server start. mlx-serve ignores
    # the field (server is bound to a single model).
    jq -nc --arg p "$prompt" --arg model "$MODEL" --argjson mt "$mt" \
        --argjson body "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"\"}],\"max_tokens\":$mt,$COMMON_BODY_FIELDS$extra}" \
        '$body | .messages[0].content = $p | .model = $model'
}

# Send one request, return "elapsed_ms|prompt_toks|completion_toks".
send_one() {
    local body="$1"
    local t0 t1 resp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(curl -sf -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body")
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ -z "$resp" ]]; then
        echo "ERR|0|0"; return
    fi
    python3 -c "
import json, sys
r = json.loads(sys.argv[1])
u = r.get('usage', {}) or {}
pt = u.get('prompt_tokens', 0)
ct = u.get('completion_tokens', 0)
print(f'{int(sys.argv[2]) - int(sys.argv[3])}|{pt}|{ct}')
" "$resp" "$t1" "$t0"
}

# Salt the prompt with a unique prefix per run so the engine's prompt cache
# can't reuse a prior request's prefix. Without this, run 2+ of an identical
# request hits the cache and reports inflated tok/s (we saw 4× inflation on
# Gemma 4 26B-A4B). The salt is small (~10 tokens) so it doesn't materially
# change prefill cost.
salted() {
    local idx="$1" base="$2"
    echo "[run ${idx} $(printf '%08x' $RANDOM$RANDOM)] ${base}"
}

# Average decode tok/s across $RUNS (run 1 dropped). Prints "tps|prompt_toks|completion_toks".
bench_cell_decode() {
    local prompt="$1" mt="$2"
    local elapsed_csv=""
    local last_pt="0" last_ct="0"
    for i in $(seq 1 "$RUNS"); do
        local body=$(build_body "$(salted "$i" "$prompt")" "$mt")
        local r=$(send_one "$body")
        local elapsed=${r%%|*}
        local rest=${r#*|}
        local pt=${rest%%|*}
        local ct=${rest#*|}
        dbg "  $LABEL/$SPEC decode run=$i elapsed=${elapsed}ms pt=$pt ct=$ct"
        last_pt="$pt"; last_ct="$ct"
        if [[ "$i" -gt 1 && "$ct" -gt 0 ]]; then
            elapsed_csv+="${elapsed},"
        fi
    done
    elapsed_csv="${elapsed_csv%,}"
    if [[ -z "$elapsed_csv" ]]; then
        echo "0|$last_pt|$last_ct"; return
    fi
    python3 -c "
e = [float(x) for x in '$elapsed_csv'.split(',') if x]
ct = $last_ct
avg_ms = sum(e)/len(e)
tps = ct / (avg_ms/1000.0) if avg_ms > 0 and ct > 0 else 0
print(f'{tps:.1f}|$last_pt|$last_ct')
"
}

# Prefill: long prompt, max_tokens=1. Returns "tps|prompt_toks|elapsed_ms".
bench_cell_prefill() {
    local elapsed_csv=""
    local last_pt="0"
    for i in $(seq 1 "$RUNS"); do
        local body=$(build_body "$(salted "$i" "$PREFILL_PROMPT")" 1)
        local r=$(send_one "$body")
        local elapsed=${r%%|*}
        local rest=${r#*|}
        local pt=${rest%%|*}
        dbg "  $LABEL/$SPEC prefill run=$i elapsed=${elapsed}ms pt=$pt"
        last_pt="$pt"
        if [[ "$i" -gt 1 && "$pt" -gt 0 ]]; then
            elapsed_csv+="${elapsed},"
        fi
    done
    elapsed_csv="${elapsed_csv%,}"
    if [[ -z "$elapsed_csv" ]]; then
        echo "0|$last_pt|0"; return
    fi
    python3 -c "
e = [float(x) for x in '$elapsed_csv'.split(',') if x]
pt = $last_pt
avg_ms = sum(e)/len(e)
tps = pt / (avg_ms/1000.0) if avg_ms > 0 and pt > 0 else 0
print(f'{tps:.1f}|$last_pt|{int(avg_ms)}')
"
}

# ── mlx-lm CLI fallback ──
# mlx_lm.server hits "There is no Stream(gpu, 0) in current thread." under MLX
# 0.31.2 (the same thread-local-stream issue we documented in CLAUDE.md). We
# fall back to `mlx_lm.generate` CLI, which prints
#   Prompt: N tokens, X tokens-per-sec
#   Generation: M tokens, Y tokens-per-sec
# directly — the per-token rates exclude model load time, so they're directly
# comparable to the HTTP-server numbers from the other engines.
mlxlm_run() {
    local prompt="$1" mt="$2"
    python3 -m mlx_lm generate \
        --model "$MODEL" \
        --prompt "$prompt" \
        --max-tokens "$mt" \
        --temp 0 \
        --seed 0 \
        --verbose True 2>&1
}

mlxlm_extract_prompt_tps() { grep -E "^Prompt:" | sed -E 's/.*: [0-9]+ tokens, ([0-9.]+) tokens-per-sec.*/\1/' | head -1; }
mlxlm_extract_decode_tps() { grep -E "^Generation:" | sed -E 's/.*: [0-9]+ tokens, ([0-9.]+) tokens-per-sec.*/\1/' | head -1; }
mlxlm_extract_prompt_tok() { grep -E "^Prompt:" | sed -E 's/Prompt: ([0-9]+) tokens.*/\1/' | head -1; }
mlxlm_extract_decode_tok() { grep -E "^Generation:" | sed -E 's/Generation: ([0-9]+) tokens.*/\1/' | head -1; }

bench_cell_mlxlm_prefill() {
    local tps_csv="" last_pt="0"
    for i in $(seq 1 "$RUNS"); do
        local out=$(mlxlm_run "$(salted "$i" "$PREFILL_PROMPT")" 1)
        local tps=$(echo "$out" | mlxlm_extract_prompt_tps)
        local pt=$(echo "$out" | mlxlm_extract_prompt_tok)
        dbg "  $LABEL/$SPEC prefill run=$i tps=$tps pt=$pt"
        last_pt="${pt:-0}"
        if [[ "$i" -gt 1 && -n "$tps" ]]; then
            tps_csv+="${tps},"
        fi
    done
    tps_csv="${tps_csv%,}"
    if [[ -z "$tps_csv" ]]; then echo "0|$last_pt|0"; return; fi
    python3 -c "
v = [float(x) for x in '$tps_csv'.split(',') if x]
print(f'{sum(v)/len(v):.1f}|$last_pt|0')"
}

bench_cell_mlxlm_decode() {
    local prompt="$1" mt="$2"
    local tps_csv="" last_pt="0" last_ct="0"
    for i in $(seq 1 "$RUNS"); do
        local out=$(mlxlm_run "$(salted "$i" "$prompt")" "$mt")
        local tps=$(echo "$out" | mlxlm_extract_decode_tps)
        local pt=$(echo "$out" | mlxlm_extract_prompt_tok)
        local ct=$(echo "$out" | mlxlm_extract_decode_tok)
        dbg "  $LABEL/$SPEC decode run=$i tps=$tps pt=$pt ct=$ct"
        last_pt="${pt:-0}"; last_ct="${ct:-0}"
        if [[ "$i" -gt 1 && -n "$tps" ]]; then
            tps_csv+="${tps},"
        fi
    done
    tps_csv="${tps_csv%,}"
    if [[ -z "$tps_csv" ]]; then echo "0|$last_pt|$last_ct"; return; fi
    python3 -c "
v = [float(x) for x in '$tps_csv'.split(',') if x]
print(f'{sum(v)/len(v):.1f}|$last_pt|$last_ct')"
}

# ── Main ──
trap 'stop_engine' EXIT INT TERM

dbg "=== $LABEL spec=$SPEC model=$MODEL port=$PORT hardware=$HARDWARE ==="

emit() { echo "$LABEL|$ENGINE|$MODEL|$SPEC|$1|$2|$3|$4|$5|$HARDWARE|$6"; }

# mlx-lm bypasses HTTP entirely — its server is broken on this MLX version.
if [[ "$ENGINE" == "mlx-lm" ]]; then
    if [[ "$DO_PREFILL" == "1" ]]; then
        out=$(bench_cell_mlxlm_prefill)
        emit "prefill" "${out%%|*}" "0" "$(echo "$out" | cut -d'|' -f2)" "1" "cli"
    fi
    if [[ "$DO_DECODE" == "1" ]]; then
        out=$(bench_cell_mlxlm_decode "$DECODE_PROMPT" "$MAX_TOKENS")
        tps=${out%%|*}; rest=${out#*|}; pt=${rest%%|*}; ct=${rest#*|}
        emit "decode" "0" "$tps" "$pt" "$ct" "cli"
    fi
    if [[ "$DO_ECHO" == "1" ]]; then
        out=$(bench_cell_mlxlm_decode "$ECHO_PROMPT" "$MAX_TOKENS")
        tps=${out%%|*}; rest=${out#*|}; pt=${rest%%|*}; ct=${rest#*|}
        emit "echo" "0" "$tps" "$pt" "$ct" "cli"
    fi
    exit 0
fi

start_engine

if [[ "$DO_PREFILL" == "1" ]]; then
    out=$(bench_cell_prefill)
    tps=${out%%|*}
    rest=${out#*|}
    pt=${rest%%|*}
    emit "prefill" "$tps" "0" "$pt" "1" ""
fi

if [[ "$DO_DECODE" == "1" ]]; then
    out=$(bench_cell_decode "$DECODE_PROMPT" "$MAX_TOKENS")
    tps=${out%%|*}
    rest=${out#*|}
    pt=${rest%%|*}
    ct=${rest#*|}
    emit "decode" "0" "$tps" "$pt" "$ct" ""
fi

if [[ "$DO_ECHO" == "1" ]]; then
    out=$(bench_cell_decode "$ECHO_PROMPT" "$MAX_TOKENS")
    tps=${out%%|*}
    rest=${out#*|}
    pt=${rest%%|*}
    ct=${rest#*|}
    emit "echo" "0" "$tps" "$pt" "$ct" ""
fi

if [[ -n "$PROMPTS_FILE" && -f "$PROMPTS_FILE" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        local_name="${line%%=*}"
        local_text="${line#*=}"
        out=$(bench_cell_decode "$local_text" "$MAX_TOKENS")
        tps=${out%%|*}
        rest=${out#*|}
        pt=${rest%%|*}
        ct=${rest#*|}
        emit "$local_name" "0" "$tps" "$pt" "$ct" ""
    done < "$PROMPTS_FILE"
fi
