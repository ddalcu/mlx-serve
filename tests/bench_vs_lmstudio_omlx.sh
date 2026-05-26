#!/bin/bash
# bench_vs_lmstudio_omlx.sh — apples-to-apples MLX-serve vs LM Studio vs oMLX
# benchmark matrix that produces the charts in docs/perf-vs-lmstudio-omlx*.png.
#
# Two model families are shipped:
#   --family gemma   Gemma 4 E2B (8bit), E4B (8bit), 31B (8bit), 26B-A4B-MoE
#                    (4bit). Compares LM Studio (MLX baseline + GGUF where
#                    available), oMLX, and MLX-serve {none, pld, drafter} where
#                    drafter uses the matching gemma-4-*-it-assistant-bf16
#                    checkpoint.
#   --family qwen36  Qwen 3.6 27B, 35B-A3B. All engines load the same
#                    standard mlx-community 4-bit MLX checkpoints (not the
#                    unsloth UD variants — UD weights are bigger on disk and
#                    less representative of what most users actually run).
#                    Compares LM Studio MLX, oMLX, and MLX-serve {none, pld},
#                    plus a GGUF baseline on the LMS side. Qwen has no
#                    Gemma-4-style drafter.
#
# Apples-to-apples controls (the same for every cell in a row):
#   - Context size: 4096 (--ctx-size on MLX-serve, --context-length on LMS).
#   - Sampling: temperature=0, top_p=1, max_tokens=128, stream=false.
#   - System prompt: none.
#   - Thinking: disabled. MLX-serve uses the `enable_thinking:false` body
#     field, which the Jinja chat template honors to render
#     `<think>\n\n</think>\n\n` before content. LM Studio silently ignores
#     `chat_template_kwargs.enable_thinking:false` on Qwen 3.6, so we use the
#     assistant-prefill workaround: send messages ending in an assistant
#     message containing the closed `<think></think>` block plus
#     `add_generation_prompt:false, continue_final_message:true`. Both paths
#     deliver identical pre-decode tokens to the model. Gemma 4 doesn't have
#     a default thinking mode, so the workaround is a no-op there.
#
# Output:
#   - CSV at $OUT (default docs/perf-vs-lmstudio-omlx-<family>.csv) with rows:
#     label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|hardware|notes
#   - To generate the chart: python3 tests/plot_vs_lmstudio_omlx.py <csv> <png> [--family <family>]
#
# Requirements:
#   - LM Studio CLI (`lms`) installed; models pre-downloaded for the chosen family.
#   - oMLX CLI (`omlx`) on PATH. If missing, omlx cells skip and the rest of
#     the matrix still runs. The script auto-flips
#     `auth.skip_api_key_verification: true` in `~/.omlx/settings.json` so the
#     bench can hit oMLX without an Authorization header.
#   - mlx-serve binary built (default ./zig-out/bin/mlx-serve, MUST be
#     -Doptimize=ReleaseFast — Debug build is 2-4× slower).
#   - jq, python3, curl on PATH.

set -uo pipefail

# ── Defaults ──
FAMILY=""
RUNS=2
MAX_TOKENS=128
CTX=4096
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PNG_OUT=""
KEEP_CSV=""
SERVER_PORT=11250
LMS_PORT=1234
# Read OMLX_PORT from ~/.omlx/settings.json if present; fall back to 11251.
if [[ -f "$HOME/.omlx/settings.json" ]]; then
    OMLX_PORT="$(python3 -c "import json,sys
try:
    print(json.load(open(sys.argv[1])).get('server',{}).get('port',11251))
except Exception:
    print(11251)" "$HOME/.omlx/settings.json" 2>/dev/null)"
    OMLX_PORT="${OMLX_PORT:-11251}"
else
    OMLX_PORT=11251
fi

usage() {
    cat <<EOF
Usage: $0 --family <gemma|qwen36> [options]

Options:
  --family NAME        Model family: 'gemma' or 'qwen36' (required)
  --out PATH           Chart PNG output path. Default is timestamped:
                       docs/perf-vs-lmstudio-omlx-<family>-YYYYMMDD-HHMMSS.png
  --keep-csv PATH      Also retain the raw CSV at this path. By default the
                       CSV is written to a temp file and deleted on exit.
  --runs N             Repeats per cell (run 1 dropped as warmup; default: 2)
  --max-tokens N       Decode budget (default: 128)
  --ctx-size N         Context size across all engines (default: 4096)
  --binary PATH        mlx-serve binary (default: ./zig-out/bin/mlx-serve)
  -h, --help           This message

Examples:
  $0 --family gemma                # writes docs/perf-vs-lmstudio-omlx-gemma-<ts>.png
  $0 --family qwen36 --runs 3      # Qwen 3.6 matrix with one extra repeat
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)     FAMILY="$2"; shift 2 ;;
        --out)        PNG_OUT="$2"; shift 2 ;;
        --keep-csv)   KEEP_CSV="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --ctx-size)   CTX="$2"; shift 2 ;;
        --binary)     BINARY="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -z "$FAMILY" ]] && { echo "Missing --family" >&2; usage; exit 1; }
[[ -x "$BINARY" ]] || { echo "Build mlx-serve first: zig build -Doptimize=ReleaseFast" >&2; exit 1; }

# Resolve repo root (this script lives in tests/, run from anywhere).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default output gets a timestamp suffix so consecutive runs don't clobber
# each other (handy when sweeping over runs / comparing tweaks). Override
# with --out PATH to pick an exact filename (e.g. one referenced by README).
TS="$(date +%Y%m%d-%H%M%S)"
[[ -z "$PNG_OUT" ]] && PNG_OUT="docs/perf-vs-lmstudio-omlx-${FAMILY}-${TS}.png"

# Raw CSV is an internal artifact of the run. Use a temp file unless the
# caller explicitly asked for it via --keep-csv.
OUT="$(mktemp -t bench_vs_lms.XXXXXX).csv"

# ── Family-specific cell definitions ──
#
# Each row: label_prefix|mlxserve_path|lms_baseline_key|lms_alt_key|drafter_dir
# - lms_baseline_key  → primary LMS baseline (UD MLX where applicable)
# - lms_alt_key       → secondary LMS variant (GGUF for Qwen, empty for Gemma)
# - drafter_dir       → Gemma 4 assistant drafter checkpoint, empty otherwise
declare -a TARGETS
case "$FAMILY" in
    gemma)
        MD="$HOME/.mlx-serve/models"
        DM="$MD/mlx-community"
        # 4-bit across the board (apples-to-apples MLX 4-bit vs GGUF Q4 vs
        # mlx-serve 4-bit). All four targets get both MLX baseline and GGUF alt.
        LMS_DIR="$HOME/.lmstudio/models"
        # LM Studio 0.3+ dropped the `model@quant` ID syntax — model IDs are
        # now the on-disk dirname (e.g. `gemma-4-e4b-it-mlx` for the MLX
        # quant, `gemma-4-e4b-it` for the GGUF). The old `gemma-4-e4b-it@4bit`
        # / `google/gemma-4-e4b` IDs return 404 against newer LM Studio.
        # The script tolerates missing rows: any TARGETS entry whose
        # `mlxserve_path` is absent skips silently.
        TARGETS=(
            "gemma4-e2b-4bit|$LMS_DIR/mlx-community/gemma-4-e2b-it-4bit|gemma-4-e2b-it-mlx|gemma-4-e2b-it|$DM/gemma-4-E2B-it-assistant-bf16"
            "gemma4-e4b-4bit|$LMS_DIR/mlx-community/gemma-4-e4b-it-4bit|gemma-4-e4b-it-mlx|gemma-4-e4b-it|$DM/gemma-4-E4B-it-assistant-bf16"
            "gemma4-31b-4bit|$LMS_DIR/mlx-community/gemma-4-31b-it-4bit|gemma-4-31b-it-mlx|gemma-4-31b-it|$DM/gemma-4-31B-it-assistant-bf16"
            "gemma4-26b-a4b-moe-4bit|$MD/gemma-4-26b-a4b-it-4bit|gemma-4-26b-a4b-it-mlx|gemma-4-26b-a4b-it|$DM/gemma-4-26B-A4B-it-assistant-bf16"
        )
        # Specs measured (per row): mlx-serve {none,pld,drafter} + omlx baseline
        # + lms_baseline + lms_alt (GGUF). Order matters: mlx-serve runs first
        # while the machine is coolest, omlx in the middle (same MLX backend so
        # it'd benefit from a fresh start too), LMS runs last so any thermal
        # throttling that builds up during the row falls on the comparison
        # engines, not on us. lms_alt rows skip silently when the row has no
        # GGUF key configured (31B, 26B-A4B currently).
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::drafter" "omlx:base:none" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        # Workaround needed? (assistant <think></think> prefill on LMS)
        LMS_THINKING_WORKAROUND=0
        ;;
    qwen36)
        LMS_DIR="$HOME/.lmstudio/models"
        # All engines load the same standard mlx-community 4-bit MLX weights
        # (not the unsloth UD variants — those are bigger on disk and we want
        # the more representative production checkpoint here).
        TARGETS=(
            "qwen36-27b|$LMS_DIR/mlx-community/Qwen3.6-27B-4bit|mlx-community/qwen3.6-27b|qwen/qwen3.6-27b|"
            "qwen36-35b-a3b|$LMS_DIR/mlx-community/Qwen3.6-35B-A3B-4bit|qwen3.6-35b-a3b@4bit|qwen/qwen3.6-35b-a3b|"
        )
        # Same ordering rule as gemma: mlx-serve first (cool machine), omlx
        # in the middle, LMS specs last so thermal throttling penalises the
        # comparison.
        SPECS=("mlx-serve::none" "mlx-serve::pld" "omlx:base:none" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        LMS_THINKING_WORKAROUND=1
        ;;
    *)
        echo "Unknown family '$FAMILY' (try gemma or qwen36)" >&2
        exit 1
        ;;
esac

# ── Test prompts (kept identical to bench_run.sh's so cross-bench numbers compare) ──
PREFILL_PROMPT="Explain the following topics in extreme detail: $(python3 -c "print(', '.join([f'topic {i} about science and technology and its impact on human civilization throughout history' for i in range(1,50)]))")"
DECODE_PROMPT="Write a detailed essay about quantum computing"
ECHO_PROMPT='Repeat the following paragraph back to me word for word, exactly as written, with no additional commentary. Then add the single sentence "End of recitation." on a new line. PARAGRAPH: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. Bright vixens jump; dozy fowl quack. Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz. Now repeat the paragraph above exactly:'
# Code-completion prompt: tests speculative-decode value on the workload the
# drafter was actually trained for. Output is fresh code so PLD's prompt-n-gram
# lookup mostly misses; whether drafter wins here is the question.
CODE_PROMPT='Implement the following Python functions. Output only the complete code in a single Python code block, no commentary.

def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using memoization."""

def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b using Euclid'\''s algorithm."""

def reverse_string(s: str) -> str:
    """Return s reversed without using slicing."""

def count_vowels(s: str) -> int:
    """Return the number of vowels (a, e, i, o, u, case-insensitive) in s."""

def is_palindrome(s: str) -> bool:
    """Return True if s is a palindrome ignoring case and non-alphanumerics."""'

# ── Body builders ──
build_body_mlx() {
    local prompt="$1" spec="$2" mt="$3"
    local epld=false edrft=false
    [[ "$spec" == "pld" ]]     && epld=true
    [[ "$spec" == "drafter" ]] && edrft=true
    jq -nc --arg p "$prompt" --argjson mt "$mt" --argjson epld "$epld" --argjson edrft "$edrft" \
        '{model:"x", messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, enable_thinking:false, enable_pld:$epld, enable_drafter:$edrft}'
}

build_body_lms() {
    local prompt="$1" model="$2" mt="$3"
    if [[ "$LMS_THINKING_WORKAROUND" == "1" ]]; then
        # Assistant-prefill `<think>\n\n</think>\n\n` so model continues in
        # content mode. Required for Qwen 3.6 since LM Studio ignores
        # chat_template_kwargs.enable_thinking on this family.
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
            '{model:$model, messages:[{role:"user",content:$p},{role:"assistant",content:"<think>\n\n</think>\n\n"}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, add_generation_prompt:false, continue_final_message:true}'
    else
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
            '{model:$model, messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, chat_template_kwargs:{enable_thinking:false}}'
    fi
}

# oMLX accepts a bare OpenAI-style body. We skip the LMS thinking workaround
# entirely — oMLX honors `chat_template_kwargs.enable_thinking` natively where
# the chat template supports it, and ignores it where it doesn't.
build_body_omlx() {
    local prompt="$1" model="$2" mt="$3"
    jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
        '{model:$model, messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, chat_template_kwargs:{enable_thinking:false}}'
}

# ── HTTP helpers ──
salted() { echo "[run-$1-$RANDOM] $2"; }

send_one() {
    local engine="$1" body="$2"
    local port
    case "$engine" in
        lmstudio)  port="$LMS_PORT" ;;
        omlx)      port="$OMLX_PORT" ;;
        *)         port="$SERVER_PORT" ;;
    esac
    local t0 t1 resp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body")
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ -z "$resp" ]]; then echo "ERR|0|0|0"; return; fi
    # Garbage-response guard: when an engine dies mid-stream (oMLX has been
    # seen to do this on later cells) curl can return a non-empty but
    # truncated body, which then crashes json.loads with an unhelpful
    # traceback. Catch it, log the raw response head, and treat as ERR
    # (the caller — run_cell — then retries the cell once).
    local parsed
    parsed=$(python3 -c "
import json,sys
try:
    r=json.loads(sys.argv[1])
except Exception as e:
    sys.stderr.write(f'  send_one: non-JSON response ({type(e).__name__}: {e}); head={sys.argv[1][:120]!r}\\n')
    sys.exit(0)
u=r.get('usage',{}) or {}
ctd=u.get('completion_tokens_details',{}) or {}
print(f\"{int(sys.argv[2])-int(sys.argv[3])}|{u.get('prompt_tokens',0)}|{u.get('completion_tokens',0)}|{ctd.get('reasoning_tokens',0)}\")
" "$resp" "$t1" "$t0")
    if [[ -z "$parsed" ]]; then echo "ERR|0|0|0"; return; fi
    echo "$parsed"
}

bench_decode() {
    local engine="$1" model="$2" spec="$3" prompt="$4" mt="$5"
    # oMLX routes requests by the basename of the model dir, not the full path.
    local omlx_model_id; omlx_model_id="$(basename "$model")"
    local elapsed_csv="" last_pt=0 last_ct=0 leaked=0
    for i in $(seq 1 "$RUNS"); do
        local body
        case "$engine" in
            mlx-serve) body=$(build_body_mlx  "$(salted "$i" "$prompt")" "$spec"  "$mt") ;;
            omlx)      body=$(build_body_omlx "$(salted "$i" "$prompt")" "$omlx_model_id" "$mt") ;;
            *)         body=$(build_body_lms  "$(salted "$i" "$prompt")" "$model" "$mt") ;;
        esac
        IFS='|' read -r elapsed pt ct rt < <(send_one "$engine" "$body")
        last_pt="$pt"; last_ct="$ct"
        [[ "$rt" -gt 0 ]] && leaked=$((leaked + rt))
        if [[ "$i" -gt 1 && "$ct" -gt 0 ]]; then
            elapsed_csv+="${elapsed},"
        fi
    done
    elapsed_csv="${elapsed_csv%,}"
    if [[ -z "$elapsed_csv" ]]; then echo "0|$last_pt|$last_ct|$leaked"; return; fi
    python3 -c "
e=[float(x) for x in '$elapsed_csv'.split(',') if x]
ct=$last_ct
avg=sum(e)/len(e)
tps=ct/(avg/1000.0) if avg>0 and ct>0 else 0
print(f'{tps:.1f}|$last_pt|$last_ct|$leaked')"
}

bench_prefill() {
    local engine="$1" model="$2" spec="$3"
    local omlx_model_id; omlx_model_id="$(basename "$model")"
    local elapsed_csv="" last_pt=0
    for i in $(seq 1 "$RUNS"); do
        local body
        case "$engine" in
            mlx-serve) body=$(build_body_mlx  "$(salted "$i" "$PREFILL_PROMPT")" "$spec"  1) ;;
            omlx)      body=$(build_body_omlx "$(salted "$i" "$PREFILL_PROMPT")" "$omlx_model_id" 1) ;;
            *)         body=$(build_body_lms  "$(salted "$i" "$PREFILL_PROMPT")" "$model" 1) ;;
        esac
        IFS='|' read -r elapsed pt ct rt < <(send_one "$engine" "$body")
        last_pt="$pt"
        if [[ "$i" -gt 1 && "$pt" -gt 0 ]]; then
            elapsed_csv+="${elapsed},"
        fi
    done
    elapsed_csv="${elapsed_csv%,}"
    if [[ -z "$elapsed_csv" ]]; then echo "0|$last_pt"; return; fi
    python3 -c "
e=[float(x) for x in '$elapsed_csv'.split(',') if x]
pt=$last_pt
avg=sum(e)/len(e)
tps=pt/(avg/1000.0) if avg>0 and pt>0 else 0
print(f'{tps:.1f}|$last_pt')"
}

# ── Engine lifecycle ──
# Disable oMLX's API-key requirement once at script start so the warmup +
# bench curls can hit it without an Authorization header. The setting persists
# in `~/.omlx/settings.json`; idempotent.
prepare_omlx_settings() {
    [[ -f "$HOME/.omlx/settings.json" ]] || return 0  # first run will create it
    python3 - "$HOME/.omlx/settings.json" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    c = json.load(f)
auth = c.setdefault("auth", {})
if not auth.get("skip_api_key_verification"):
    auth["skip_api_key_verification"] = True
    with open(p, "w") as f:
        json.dump(c, f, indent=2)
PY
}

stop_all_engines() {
    pkill -9 -x mlx-serve 2>/dev/null
    # oMLX launches as `python3 -m omlx.cli serve …` — match by `omlx.cli`.
    pkill -9 -f "omlx.cli" 2>/dev/null
    # Belt-and-suspenders: clear known ports if anything survived.
    for p in "$SERVER_PORT" "$OMLX_PORT"; do
        local pids; pids="$(lsof -ti:"$p" 2>/dev/null)"
        [[ -n "$pids" ]] && echo "$pids" | xargs -r kill -9 2>/dev/null
    done
    # Wait for both ports to actually free up — pkill returns before the kernel
    # reaps the process, and a quick relaunch can race the old socket holding
    # the port (which then causes the new engine to die on bind). Plus give the
    # kernel time to reclaim the prior model's MLX buffers (5-20 GB) before the
    # next engine allocates — when this is too short, oMLX in particular has
    # been observed to die mid-prefill on the next cell.
    local waited=0
    while (( waited < 30 )); do
        local busy=0
        for p in "$SERVER_PORT" "$OMLX_PORT" "$LMS_PORT"; do
            if lsof -ti:"$p" >/dev/null 2>&1; then busy=1; break; fi
        done
        (( busy == 0 )) && break
        sleep 1; waited=$((waited+1))
    done
    sleep 5
}

start_engine() {
    local engine="$1" model_or_path="$2" spec="$3" drafter="$4"
    ENGINE_PID=""   # global the caller polls to detect mid-cell death
    stop_all_engines
    case "$engine" in
        mlx-serve)
            local extra=""
            case "$spec" in
                none)    extra="--no-pld" ;;
                pld)     extra="" ;;
                drafter) [[ -z "$drafter" ]] && { echo "  drafter spec missing --drafter dir" >&2; return 1; }
                         extra="--no-pld --drafter $drafter" ;;
            esac
            "$BINARY" --model "$model_or_path" --serve --port "$SERVER_PORT" \
                --ctx-size "$CTX" --log-level info $extra >/tmp/bench_vs_lms_engine.log 2>&1 &
            local pid=$!
            ENGINE_PID="$pid"
            for i in $(seq 1 240); do
                curl -sf "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1 && return 0
                sleep 0.5
                kill -0 "$pid" 2>/dev/null || { echo "  mlx-serve died" >&2; return 1; }
            done
            return 1
            ;;
        lmstudio)
            lms server start --port "$LMS_PORT" >/dev/null 2>&1
            for i in $(seq 1 60); do
                curl -sf "http://127.0.0.1:$LMS_PORT/v1/models" >/dev/null 2>&1 && break
                sleep 0.5
            done
            lms unload --all >/dev/null 2>&1
            # `lms load` is unreliable on some LM Studio releases (silently hangs
            # for many minutes). HTTP JIT-load via /v1/chat/completions is the
            # supported path: LM Studio loads the model on first request when
            # `Just-In-Time Model Loading` is enabled (default in 0.4.x). The
            # warmup curl with a long timeout serves as both load-trigger and
            # health probe.
            local warmup_body
            warmup_body=$(jq -nc --arg model "$model_or_path" '{model:$model,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false}')
            if ! curl -sf -m 600 -X POST "http://127.0.0.1:$LMS_PORT/v1/chat/completions" \
                -H "Content-Type: application/json" -d "$warmup_body" >/dev/null 2>&1; then
                echo "  lms HTTP JIT-load failed for $model_or_path (timed out at 600s)" >&2
                return 1
            fi
            ;;
        omlx)
            # oMLX serves a `--model-dir` (parent) and routes requests by
            # subdir name (= basename of the model path). Same MLX-format
            # weights as mlx-serve, no conversion needed.
            local model_dir model_id
            model_dir="$(dirname "$model_or_path")"
            model_id="$(basename "$model_or_path")"
            # shellcheck disable=SC2086
            omlx serve --model-dir "$model_dir" --port "$OMLX_PORT" \
                >/tmp/bench_vs_lms_omlx.log 2>&1 &
            local pid=$!
            ENGINE_PID="$pid"
            for i in $(seq 1 240); do
                curl -sf "http://127.0.0.1:$OMLX_PORT/v1/models" >/dev/null 2>&1 && break
                sleep 0.5
                kill -0 "$pid" 2>/dev/null || { echo "  omlx died (tail of /tmp/bench_vs_lms_omlx.log:)" >&2; tail -n 15 /tmp/bench_vs_lms_omlx.log >&2; return 1; }
                [[ "$i" -eq 240 ]] && { echo "  omlx /v1/models never came up" >&2; return 1; }
            done
            # JIT-load warmup so the first timed request doesn't pay the
            # model-load cost.
            local warmup_body
            warmup_body=$(jq -nc --arg model "$model_id" '{model:$model,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false}')
            if ! curl -sf -m 600 -X POST "http://127.0.0.1:$OMLX_PORT/v1/chat/completions" \
                -H "Content-Type: application/json" -d "$warmup_body" >/dev/null 2>&1; then
                echo "  omlx warmup failed for $model_id (check /tmp/bench_vs_lms_omlx.log)" >&2
                return 1
            fi
            ;;
    esac
}

# ── Hardware tag (Phase B) ──
# Tag every row with the host's chip + RAM so CSVs from different Macs can be
# merged/diffed without losing provenance.
HARDWARE_TAG="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
HARDWARE_TAG="${HARDWARE_TAG// /-}"
if [[ "$(uname)" == "Darwin" ]]; then
    _ram_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))
else
    _ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
fi
HARDWARE_TAG="${HARDWARE_TAG}-${_ram_gb}gb"
unset _ram_gb

# ── Output emission ──
echo "label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|hardware|notes" > "$OUT"
emit() {
    local label="$1" engine="$2" model="$3" spec="$4" prompt_kind="$5" pf="$6" dc="$7" pt="$8" ct="$9" notes="${10}"
    echo "${label}|${engine}|${model}|${spec}|${prompt_kind}|${pf}|${dc}|${pt}|${ct}|${HARDWARE_TAG}|${notes}" | tee -a "$OUT"
}

run_cell() {
    local label_prefix="$1" engine="$2" model_or_path="$3" spec="$4" drafter="$5" notes="$6"
    local label="${label_prefix}"
    echo "  >> $label / $engine / $spec" >&2
    if ! start_engine "$engine" "$model_or_path" "$spec" "$drafter"; then
        echo "  SKIP $label" >&2
        return
    fi

    local out tps pt ct rt cell_notes retry_out prompt kind

    # engine_alive: true unless we have a tracked PID and it's dead. LM Studio
    # doesn't expose a PID we can watch (`lms server start` daemonizes), so we
    # treat ENGINE_PID="" as "always alive" and skip the mid-cell retry on
    # that path.
    engine_alive() {
        [[ -z "$ENGINE_PID" ]] && return 0
        kill -0 "$ENGINE_PID" 2>/dev/null
    }

    # retry_if_bad: when the prior call returned 0 tps, retry once. If the
    # tracked engine PID is dead (oMLX has been seen to die mid-prefill on
    # later cells) we restart it first; otherwise we just re-send the body
    # (covers transient curl/JSON hiccups, the LMS-without-PID case, and any
    # engine that's alive but returned garbage). The retry is bounded to one
    # round so a permanently-broken engine costs at most one extra request.
    retry_if_bad() {
        local kind="$1" tps_in="$2"; shift 2  # remaining: bench fn args
        if [[ "$tps_in" != "0" ]]; then echo ""; return; fi
        if ! engine_alive; then
            echo "  $engine died mid-cell ($kind) — restarting and retrying once" >&2
            if ! start_engine "$engine" "$model_or_path" "$spec" "$drafter"; then
                echo "  retry: start_engine failed; giving up on $label" >&2
                return
            fi
        else
            echo "  $engine returned 0 tps ($kind) — retrying once" >&2
        fi
        if [[ "$kind" == "prefill" ]]; then
            bench_prefill "$engine" "$model_or_path" "$spec"
        else
            bench_decode  "$engine" "$model_or_path" "$spec" "$1" "$2"
        fi
    }

    out=$(bench_prefill "$engine" "$model_or_path" "$spec")
    IFS='|' read -r tps pt <<<"$out"
    retry_out=$(retry_if_bad "prefill" "$tps")
    [[ -n "$retry_out" ]] && { out="$retry_out"; IFS='|' read -r tps pt <<<"$out"; }
    emit "$label" "$engine" "$model_or_path" "$spec" "prefill" "$tps" "0" "$pt" "1" "$notes"

    for kind in decode echo code; do
        case "$kind" in
            decode) prompt="$DECODE_PROMPT" ;;
            echo)   prompt="$ECHO_PROMPT" ;;
            code)   prompt="$CODE_PROMPT" ;;
        esac
        out=$(bench_decode "$engine" "$model_or_path" "$spec" "$prompt" "$MAX_TOKENS")
        IFS='|' read -r tps pt ct rt <<<"$out"
        retry_out=$(retry_if_bad "$kind" "$tps" "$prompt" "$MAX_TOKENS")
        [[ -n "$retry_out" ]] && { out="$retry_out"; IFS='|' read -r tps pt ct rt <<<"$out"; }
        cell_notes="$notes"
        [[ "$rt" -gt 0 ]] && cell_notes="${cell_notes:+$cell_notes,}thinking_leaked=$rt"
        emit "$label" "$engine" "$model_or_path" "$spec" "$kind" "0" "$tps" "$pt" "$ct" "$cell_notes"
    done
}

cleanup() {
    stop_all_engines
    lms unload --all >/dev/null 2>&1 || true
    # Remove the temp CSV unless the caller asked to keep it.
    if [[ -z "$KEEP_CSV" && -n "$OUT" && -f "$OUT" ]]; then
        rm -f "$OUT"
    fi
}
trap cleanup EXIT INT TERM

# Pre-flight: oMLX availability + auth-disable. Missing `omlx` doesn't abort
# the run — the omlx cells just skip.
HAS_OMLX=0
if command -v omlx >/dev/null 2>&1; then
    HAS_OMLX=1
    prepare_omlx_settings
else
    echo "(omlx not on PATH — omlx cells will be skipped)" >&2
fi

for row in "${TARGETS[@]}"; do
    IFS='|' read -r logical mlxserve_path lms_baseline lms_alt drafter <<<"$row"
    [[ -d "$mlxserve_path" ]] || { echo "SKIP missing $mlxserve_path" >&2; continue; }

    for spec_entry in "${SPECS[@]}"; do
        IFS=':' read -r engine variant spec <<<"$spec_entry"
        case "$engine|$variant" in
            "lmstudio|lms_baseline")
                run_cell "${logical}/lmstudio-baseline/${spec}"  "lmstudio"  "$lms_baseline"  "$spec" "" ""
                ;;
            "lmstudio|lms_alt")
                [[ -z "$lms_alt" ]] && continue
                run_cell "${logical}/lmstudio-alt/${spec}"       "lmstudio"  "$lms_alt"       "$spec" "" ""
                ;;
            "omlx|base")
                [[ "$HAS_OMLX" -eq 1 ]] || { echo "  SKIP ${logical}/omlx/${spec} (omlx not on PATH)" >&2; continue; }
                run_cell "${logical}/omlx/${spec}"               "omlx"      "$mlxserve_path" "$spec" "" ""
                ;;
            "mlx-serve|"|"mlx-serve|*")
                # Skip drafter cell when no drafter dir is present (Qwen).
                if [[ "$spec" == "drafter" && ( -z "$drafter" || ! -d "$drafter" ) ]]; then
                    continue
                fi
                run_cell "${logical}/mlx-serve/${spec}"          "mlx-serve" "$mlxserve_path" "$spec" "$drafter" ""
                ;;
        esac
    done
done

stop_all_engines
lms unload --all >/dev/null 2>&1 || true

# Render the chart (only artifact most users want).
mkdir -p "$(dirname "$PNG_OUT")"
python3 "$SCRIPT_DIR/plot_vs_lmstudio_omlx.py" "$OUT" "$PNG_OUT" --family "$FAMILY"

# Optionally retain the raw CSV; otherwise it's a tempfile and gets cleaned
# up by the EXIT trap below.
if [[ -n "$KEEP_CSV" ]]; then
    mkdir -p "$(dirname "$KEEP_CSV")"
    cp "$OUT" "$KEEP_CSV"
    echo "CSV retained at $KEEP_CSV"
fi

echo
echo "=== chart written to $PNG_OUT ==="
