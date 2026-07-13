#!/bin/bash
# bench.sh — unified mlx-serve performance bench.
#
# Default run is mlx-serve only across {none,pld,drafter} × prefill/decode/echo/code
# prompts (fast dev-loop iteration). Pass --lmstudio and/or --omlx to add the
# apples-to-apples comparison cells that produce charts in
# docs/perf-vs-lmstudio-omlx*.png. Pass --concurrent N to also emit batched
# throughput rows (folded from the old bench_concurrent.py).
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
ONLY=""   # substring filter on the logical target name (e.g. "e2b", "31b"); empty = all
SPECS_ONLY="" # substring filter on the spec entry (e.g. "mlx-serve", "lmstudio"); empty = all
RAW=0     # when 1, drop the PLD/drafter mlx-serve cells (raw apples-to-apples only)
THINKING=0 # when 1, enable reasoning on every engine (fair same-workload Qwen comparison)
RUNS=2
MAX_TOKENS=128
CTX=4096
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PNG_OUT=""
KEEP_CSV=""
SERVER_PORT=11250
LMS_PORT=1234
# Comparison engines are opt-in. Default is mlx-serve cells only — that's the
# fast "did my change move perf" loop. Pass --lmstudio and/or --omlx to enable
# the engine-comparison cells (slower; each adds boot+warmup time per row).
INCLUDE_LMSTUDIO=0
INCLUDE_OMLX=0
# Concurrent throughput mode (folded from the old bench_concurrent.py). When >1,
# starts mlx-serve with --max-concurrent N and emits an extra `decode_c<N>`
# row per cell that fires N parallel requests; the row's tok/s is the aggregate
# rate (sum of completion_tokens / wall). Compares cleanly against the single-
# request decode row above it to see the batching speedup.
CONCURRENT=0
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

The default run measures mlx-serve only (MLX safetensors + GGUF where vendored)
across {none, pld, drafter} and the prefill/decode/echo/code prompts. Add
--lmstudio and/or --omlx to include the comparison engines.

Options:
  --family NAME        Model family: 'gemma' or 'qwen36' (required)
  --only SUBSTR        Only run targets whose logical name contains SUBSTR
  --specs SUBSTR       Only run spec entries containing SUBSTR (e.g.
                       'mlx-serve' or 'lmstudio') — split long rows across
                       invocations, then merge the --keep-csv slices
  --lmstudio           Include LM Studio cells (MLX baseline + GGUF alt where
                       configured). Requires \`lms\` CLI; LM Studio handles JIT
                       model load on the warmup curl.
  --omlx               Include oMLX cells. Requires \`omlx\` on PATH; silently
                       skipped if missing.
  --concurrent N       Also emit a \`decode_c<N>\` row per mlx-serve cell. The
                       server is started with --max-concurrent N and N parallel
                       /v1/chat/completions are fired; the row's tok/s is the
                       aggregate rate. Default 0 (off).
  --out PATH           Chart PNG output path. Default is timestamped:
                       docs/perf-vs-lmstudio-omlx-<family>-YYYYMMDD-HHMMSS.png
                       The chart is skipped when no comparison engines are
                       enabled (a single-engine bar chart isn't useful).
  --keep-csv PATH      Also retain the raw CSV at this path. By default the
                       CSV is written to a temp file and deleted on exit.
  --runs N             Repeats per cell (run 1 dropped as warmup; default: 2)
  --max-tokens N       Decode budget (default: 128)
  --ctx-size N         Context size across all engines (default: 4096)
  --binary PATH        mlx-serve binary (default: ./zig-out/bin/mlx-serve)
  -h, --help           This message

Examples:
  $0 --family gemma                            # mlx-serve only (fast iteration)
  $0 --family gemma --lmstudio --omlx          # full apples-to-apples chart
  $0 --family qwen36 --concurrent 2            # add 2-way batched row
  $0 --family gemma --lmstudio --keep-csv x.csv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)     FAMILY="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --specs)      SPECS_ONLY="$2"; shift 2 ;;
        --raw)        RAW=1; shift ;;
        --thinking)   THINKING=1; shift ;;
        --lmstudio)   INCLUDE_LMSTUDIO=1; shift ;;
        --omlx)       INCLUDE_OMLX=1; shift ;;
        --concurrent) CONCURRENT="$2"; shift 2 ;;
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
# Each row: label_prefix|mlxserve_mlx_path|lms_baseline|lms_alt|drafter_dir|mlxserve_gguf_path
# - mlxserve_mlx_path   → MLX safetensors dir for the mlx-serve cells
# - lms_baseline        → primary LMS model id (MLX baseline where applicable)
# - lms_alt             → secondary LMS variant (GGUF id)
# - drafter_dir         → Gemma 4 assistant drafter checkpoint, empty otherwise
# - mlxserve_gguf_path  → .gguf file (or dir containing one) for the mlx-serve
#                         GGUF cell. Same artifact LMS loads for `lms_alt` so
#                         the head-to-head is apples-to-apples. Empty/missing
#                         skips the mlx-serve-gguf cell for this row.
declare -a TARGETS
case "$FAMILY" in
    gemma)
        MD="$HOME/.mlx-serve/models"
        DM="$MD/mlx-community"
        # 4-bit across the board (apples-to-apples MLX 4-bit vs GGUF Q4 vs
        # mlx-serve 4-bit). All four targets get both MLX baseline and GGUF alt.
        LMS_DIR="$HOME/.lmstudio/models"
        # Vendored GGUFs — same files LM Studio loads under the hood. Default
        # to the LM Studio community dir under $LMS_DIR; override the per-row
        # path in TARGETS if your GGUFs live elsewhere.
        GGUF_DIR="$LMS_DIR/lmstudio-community"
        # LM Studio 0.3+ model IDs are the upstream HF org/name. Verify with
        # `curl -sf http://127.0.0.1:1234/v1/models` — the MLX baseline lives
        # under `mlx-community/<name>`, the GGUF alt under `google/<name>` (or
        # `<name>` without an org for some entries). Older `*-it-mlx` /
        # `<name>` fuzzy IDs no longer JIT-load and timed out the bench
        # silently. The script tolerates missing rows: any TARGETS entry
        # whose `mlxserve_path` is absent skips silently.
        TARGETS=(
            "gemma4-e2b-4bit|$LMS_DIR/mlx-community/gemma-4-e2b-it-4bit|mlx-community/gemma-4-e2b-it|google/gemma-4-e2b|$DM/gemma-4-E2B-it-assistant-bf16|$GGUF_DIR/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf"
            "gemma4-e4b-4bit|$LMS_DIR/mlx-community/gemma-4-e4b-it-4bit|mlx-community/gemma-4-e4b-it|google/gemma-4-e4b|$DM/gemma-4-E4B-it-assistant-bf16|$GGUF_DIR/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
            "gemma4-12b-4bit|$LMS_DIR/mlx-community/gemma-4-12B-it-4bit|mlx-community/gemma-4-12b-it|google/gemma-4-12b|$DM/gemma-4-12B-it-assistant-bf16|$GGUF_DIR/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
            "gemma4-12b-qat-4bit|$LMS_DIR/mlx-community/gemma-4-12b-it-qat-4bit|mlx-community/gemma-4-12b-it-qat|google/gemma-4-12b-qat|$DM/gemma-4-12B-it-assistant-bf16|$GGUF_DIR/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"
            # 26B-A4B row: the QAT 4-bit checkpoint (the non-qat MLX dir was
            # retired). Both engines load the SAME weights from the LMS dir;
            # LM Studio advertises this side-loaded dir with a bare id (no
            # mlx-community/ prefix) — verified via /v1/models. The GGUF alt
            # stays the non-QAT Q4_K_M (same file on both engines).
            "gemma4-26b-a4b-moe-qat-4bit|$LMS_DIR/mlx-community/gemma-4-26B-A4B-it-qat-4bit|gemma-4-26b-a4b-it-qat|google/gemma-4-26b-a4b|$DM/gemma-4-26B-A4B-it-assistant-bf16|$GGUF_DIR/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"
            "gemma4-31b-4bit|$LMS_DIR/mlx-community/gemma-4-31b-it-4bit|mlx-community/gemma-4-31b-it|google/gemma-4-31b|$DM/gemma-4-31B-it-assistant-bf16|$GGUF_DIR/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
        )
        # Specs measured (per row): mlx-serve {none,pld,drafter} + mlx-serve
        # GGUF (none — PLD/drafter are MLX-only and silently no-op on the
        # llama.cpp path) + omlx baseline + lms_baseline + lms_alt (GGUF).
        # Order matters: mlx-serve MLX first (cool machine), mlx-serve GGUF
        # next, omlx in the middle, LMS specs last so any thermal throttling
        # that builds up during the row falls on the comparison engines, not
        # on us. lms_alt rows skip silently when the row has no GGUF key
        # configured (31B, 26B-A4B currently).
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::drafter" "mlx-serve:alt:none" "omlx:base:none" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        # Gemma 4 has no thinking mode; the LMS workaround is a no-op.
        LMS_THINKING_WORKAROUND=0
        ;;
    qwen36)
        LMS_DIR="$HOME/.lmstudio/models"
        GGUF_DIR="$LMS_DIR/lmstudio-community"
        # All engines load the same standard mlx-community 4-bit MLX weights
        # (not the unsloth UD variants — those are bigger on disk and we want
        # the more representative production checkpoint here). GGUFs are the
        # lmstudio-community Q4_K_M vendored builds.
        TARGETS=(
            # LM Studio advertises these side-loaded dirs with bare ids (no
            # mlx-community/ prefix) — verified via /v1/models on 0.4.15.
            "qwen36-27b|$LMS_DIR/mlx-community/Qwen3.6-27B-4bit|qwen3.6-27b|qwen/qwen3.6-27b||$GGUF_DIR/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
            "qwen36-35b-a3b|$LMS_DIR/mlx-community/Qwen3.6-35B-A3B-4bit|qwen3.6-35b-a3b|qwen/qwen3.6-35b-a3b||$GGUF_DIR/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
        )
        # Same ordering rule as gemma: mlx-serve MLX first (cool machine),
        # mlx-serve GGUF next, omlx in the middle, LMS specs last so any
        # thermal throttling penalises the comparison engines, not us.
        # `mtp` = the native multi-token-prediction sidecar shipped inside
        # these checkpoints (mtp/weights.safetensors / model-mtp.safetensors).
        # It is mlx-serve's out-of-the-box default on the dense 27B, so it
        # gets its own labeled cell; `none`/`pld` pass --no-mtp so their
        # labels stay truthful (priority is MTP > PLD when the sidecar is
        # loaded — without --no-mtp both cells would silently measure MTP).
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::mtp" "mlx-serve:alt:none" "omlx:base:none" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        # Qwen 3.6's chat template auto-activates `<think>` mode; LM Studio
        # ignores `chat_template_kwargs.enable_thinking:false`, so build_body_lms
        # uses the stacked workaround when this flag is on.
        LMS_THINKING_WORKAROUND=1
        ;;
    *)
        echo "Unknown family '$FAMILY' (try gemma or qwen36)" >&2
        exit 1
        ;;
esac

# --raw: drop the speculative-decode cells (pld/drafter); keep only the raw
# apples-to-apples comparison (mlx-serve MLX `none`, mlx-serve GGUF, omlx, LMS).
if [[ "$RAW" -eq 1 ]]; then
    _raw_specs=()
    for _s in "${SPECS[@]}"; do
        [[ "$_s" == *":pld" || "$_s" == *":drafter" || "$_s" == *":mtp" ]] && continue
        _raw_specs+=("$_s")
    done
    SPECS=("${_raw_specs[@]}")
fi

# ── Test prompts (identical wording across engines so cross-bench numbers compare) ──
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
# Thinking is suppressed across all engines so prefill/decode/echo/code cells
# measure the same workload. Thinking-on produces measurement asymmetry: LMS
# excludes reasoning tokens from max_tokens while mlx-serve doesn't, so the
# two engines end up decoding very different amounts of tokens for the same
# nominal max_tokens. Gemma 4 has no thinking mode (enable_thinking is a
# no-op). Qwen 3.6 needs the stacked workaround in build_body_lms — LMS
# silently ignores chat_template_kwargs.enable_thinking:false on this family.
build_body_mlx() {
    local prompt="$1" spec="$2" mt="$3"
    local epld=false edrft=false emtp=false think=false
    [[ "$spec" == "pld" ]]     && epld=true
    [[ "$spec" == "drafter" ]] && edrft=true
    # enable_mtp:true is the per-request opt-in for MoE targets (default-off
    # there); on dense targets it matches the default. none/pld cells rely on
    # the server-side --no-mtp instead so this stays a plain default-false.
    [[ "$spec" == "mtp" ]]     && emtp=true
    [[ "$THINKING" == "1" ]]   && think=true
    jq -nc --arg p "$prompt" --argjson mt "$mt" --argjson epld "$epld" --argjson edrft "$edrft" --argjson emtp "$emtp" --argjson think "$think" \
        '{model:"x", messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, enable_thinking:$think, enable_pld:$epld, enable_drafter:$edrft, enable_mtp:$emtp}'
}

build_body_lms() {
    local prompt="$1" model="$2" mt="$3"
    if [[ "$THINKING" == "1" ]]; then
        # Thinking ON: let LM Studio reason normally (same workload as the other
        # engines). tok/s is measured over (answer+reasoning)/wall downstream.
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
            '{model:$model, messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, chat_template_kwargs:{enable_thinking:true}}'
        return
    fi
    if [[ "$LMS_THINKING_WORKAROUND" == "1" ]]; then
        # Thinking-suppression for Qwen 3.6 on LM Studio. The old
        # assistant-prefilled `<think></think>` + continue_final_message trick
        # DID suppress thinking but made LMS ~5× slower (probably re-running
        # the chat template on every decode step). Instead we use a lighter
        # combo: a system message + Qwen's native `/no_think` suffix +
        # chat_template_kwargs. Even when the model still thinks, LMS double-
        # reports the work in `completion_tokens` (alongside reasoning_tokens),
        # so dividing by wall gives the honest engine decode rate.
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" '{
            model: $model,
            messages: [
                {role: "system", content: "Respond directly. Do not emit any <think> or </think> tokens. Provide the final answer immediately."},
                {role: "user", content: ($p + "  /no_think")}
            ],
            max_tokens: $mt, temperature: 0.0, top_p: 1.0, stream: false,
            chat_template_kwargs: {enable_thinking: false}
        }'
    else
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
            '{model:$model, messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, chat_template_kwargs:{enable_thinking:false}}'
    fi
}

# oMLX accepts a bare OpenAI-style body; it honors chat_template_kwargs natively.
build_body_omlx() {
    local prompt="$1" model="$2" mt="$3"
    local think=false; [[ "$THINKING" == "1" ]] && think=true
    jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" --argjson think "$think" \
        '{model:$model, messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, chat_template_kwargs:{enable_thinking:$think}}'
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
    # Returns: elapsed_ms|prompt_tokens|completion_tokens|reasoning_tokens
    # Thinking is suppressed across all engines; reasoning_tokens > 0 means
    # the suppression LEAKED on that cell, and the row should be treated as
    # unreliable (tok/s based on completion_tokens alone undercounts what the
    # engine actually decoded).
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
ct=int(u.get('completion_tokens') or 0)
rt=int(ctd.get('reasoning_tokens') or 0)
print(f\"{int(sys.argv[2])-int(sys.argv[3])}|{u.get('prompt_tokens',0)}|{ct}|{rt}\")
" "$resp" "$t1" "$t0")
    if [[ -z "$parsed" ]]; then echo "ERR|0|0|0"; return; fi
    echo "$parsed"
}

bench_decode() {
    local engine="$1" model="$2" spec="$3" prompt="$4" mt="$5"
    # oMLX routes requests by the basename of the model dir, not the full path.
    local omlx_model_id; omlx_model_id="$(basename "$model")"
    local port
    case "$engine" in lmstudio) port="$LMS_PORT";; omlx) port="$OMLX_PORT";; *) port="$SERVER_PORT";; esac
    # Decode rate is measured from the STREAM (tests/_decode_stream.py): it counts
    # actual content+reasoning delta pieces and times first->last token, so it does
    # NOT depend on the server's usage/token accounting (LM Studio reports those
    # inconsistently once reasoning is involved) and it excludes prefill. This is
    # the only cross-engine-fair way to compare tok/s when some engines emit
    # reasoning. PLD/drafter accepted tokens stream one delta each, so their
    # speedup shows up correctly in the rate.
    local rates=() last_n=0 leaked=0
    for i in $(seq 1 "$RUNS"); do
        local body
        case "$engine" in
            mlx-serve) body=$(build_body_mlx  "$(salted "$i" "$prompt")" "$spec"  "$mt") ;;
            omlx)      body=$(build_body_omlx "$(salted "$i" "$prompt")" "$omlx_model_id" "$mt") ;;
            *)         body=$(build_body_lms  "$(salted "$i" "$prompt")" "$model" "$mt") ;;
        esac
        local out; out=$(printf '%s' "$body" | python3 "$SCRIPT_DIR/_decode_stream.py" "http://127.0.0.1:$port/v1/chat/completions" 240)
        IFS='|' read -r rate ntok rn <<<"$out"
        last_n="$ntok"; [[ "${rn:-0}" -gt 0 ]] 2>/dev/null && leaked=$((leaked + rn))
        if [[ "$i" -gt 1 && "$rate" != "ERR" && "$rate" != "0" ]]; then rates+=("$rate"); fi
    done
    if [[ ${#rates[@]} -eq 0 ]]; then echo "0|0|$last_n|$leaked"; return; fi
    python3 -c "
r=[float(x) for x in '${rates[*]}'.split()]
print(f'{sum(r)/len(r):.1f}|0|$last_n|$leaked')"
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

# ── Concurrent decode (folded from bench_concurrent.py) ──
# Fires $n parallel /v1/chat/completions and reports aggregate throughput
# (sum of completion_tokens across all requests, divided by total wall time).
# The single-request decode row above this in the CSV gives the per-request
# baseline; the speedup = aggregate_tps / single_request_tps shows whether
# the engine's batched scheduling actually helps for this workload.
bench_decode_concurrent() {
    local engine="$1" model="$2" spec="$3" prompt="$4" mt="$5" n="$6"
    local omlx_model_id; omlx_model_id="$(basename "$model")"
    local port
    case "$engine" in
        lmstudio) port="$LMS_PORT" ;;
        omlx)     port="$OMLX_PORT" ;;
        *)        port="$SERVER_PORT" ;;
    esac
    local outdir; outdir=$(mktemp -d -t bench_conc.XXXXXX)
    local pids=() i
    local t0; t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    for i in $(seq 1 "$n"); do
        local body
        case "$engine" in
            mlx-serve) body=$(build_body_mlx  "$(salted "c$i" "$prompt")" "$spec"  "$mt") ;;
            omlx)      body=$(build_body_omlx "$(salted "c$i" "$prompt")" "$omlx_model_id" "$mt") ;;
            *)         body=$(build_body_lms  "$(salted "c$i" "$prompt")" "$model" "$mt") ;;
        esac
        (
            curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
                -H "Content-Type: application/json" -d "$body" > "$outdir/$i.json" 2>/dev/null
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    local t1; t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    local result
    result=$(python3 -c "
import json, glob, sys
elapsed_ms = $t1 - $t0
tot_ct = tot_pt = 0
ok = 0
for f in sorted(glob.glob('$outdir/*.json')):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    u = r.get('usage') or {}
    ct = int(u.get('completion_tokens') or 0)
    pt = int(u.get('prompt_tokens') or 0)
    if ct > 0:
        ok += 1
    tot_ct += ct
    tot_pt += pt
tps = (tot_ct / (elapsed_ms / 1000.0)) if elapsed_ms > 0 and tot_ct > 0 else 0
print(f'{tps:.1f}|{tot_pt}|{tot_ct}|0|{ok}')")
    rm -rf "$outdir"
    echo "$result"
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
                none)    extra="--no-pld --no-mtp" ;;
                pld)     extra="--no-mtp" ;;
                mtp)     extra="--no-pld" ;;
                drafter) [[ -z "$drafter" ]] && { echo "  drafter spec missing --drafter dir" >&2; return 1; }
                         extra="--no-pld --drafter $drafter" ;;
            esac
            # --max-concurrent N enables N-way batched scheduling on dense archs.
            # Always pass the flag — N=1 (default) matches single-slot behavior.
            local mc_arg=""
            [[ "$CONCURRENT" -gt 1 ]] && mc_arg="--max-concurrent $CONCURRENT"
            "$BINARY" --model "$model_or_path" --serve --port "$SERVER_PORT" \
                --ctx-size "$CTX" --log-level info $extra $mc_arg >/tmp/bench_vs_lms_engine.log 2>&1 &
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

    # Concurrent throughput pass: N parallel decode requests, aggregate tok/s.
    # The single-request `decode` row above is the per-request baseline;
    # `decode_c<N>` shows whether batched scheduling actually multiplies it.
    if [[ "$CONCURRENT" -gt 1 ]]; then
        out=$(bench_decode_concurrent "$engine" "$model_or_path" "$spec" "$DECODE_PROMPT" "$MAX_TOKENS" "$CONCURRENT")
        IFS='|' read -r tps pt ct rt ok <<<"$out"
        cell_notes="$notes"
        cell_notes="${cell_notes:+$cell_notes,}concurrent=$CONCURRENT,ok=$ok"
        emit "$label" "$engine" "$model_or_path" "$spec" "decode_c${CONCURRENT}" "0" "$tps" "$pt" "$ct" "$cell_notes"
    fi
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

# Pre-flight: oMLX availability + auth-disable. Only checked when --omlx is
# passed; without the flag, oMLX cells are skipped wholesale regardless of
# whether the CLI is installed.
HAS_OMLX=0
if [[ "$INCLUDE_OMLX" -eq 1 ]]; then
    if command -v omlx >/dev/null 2>&1; then
        HAS_OMLX=1
        prepare_omlx_settings
    else
        echo "--omlx passed but omlx not on PATH; cells will be skipped" >&2
    fi
fi
[[ "$INCLUDE_LMSTUDIO" -eq 0 && "$INCLUDE_OMLX" -eq 0 ]] && \
    echo "(mlx-serve only — pass --lmstudio and/or --omlx to add comparison engines)" >&2

for row in "${TARGETS[@]}"; do
    IFS='|' read -r logical mlxserve_path lms_baseline lms_alt drafter mlxserve_gguf_path <<<"$row"
    [[ -n "$ONLY" && "$logical" != *"$ONLY"* ]] && { echo "SKIP $logical (--only $ONLY)" >&2; continue; }
    [[ -d "$mlxserve_path" ]] || { echo "SKIP missing $mlxserve_path" >&2; continue; }

    for spec_entry in "${SPECS[@]}"; do
        # --specs: substring filter on the spec entry ("mlx-serve" matches all
        # four mlx-serve cells incl. the GGUF alt; "lmstudio" the two LMS
        # cells). Lets a long row be split across invocations so each stays
        # under an external per-call time cap; merge the --keep-csv slices.
        [[ -n "$SPECS_ONLY" && "$spec_entry" != *"$SPECS_ONLY"* ]] && continue
        IFS=':' read -r engine variant spec <<<"$spec_entry"
        case "$engine|$variant" in
            "lmstudio|lms_baseline")
                [[ "$INCLUDE_LMSTUDIO" -eq 1 ]] || continue
                run_cell "${logical}/lmstudio-baseline/${spec}"  "lmstudio"  "$lms_baseline"  "$spec" "" ""
                ;;
            "lmstudio|lms_alt")
                [[ "$INCLUDE_LMSTUDIO" -eq 1 ]] || continue
                [[ -z "$lms_alt" ]] && continue
                run_cell "${logical}/lmstudio-alt/${spec}"       "lmstudio"  "$lms_alt"       "$spec" "" ""
                ;;
            "omlx|base")
                [[ "$INCLUDE_OMLX" -eq 1 ]] || continue
                [[ "$HAS_OMLX" -eq 1 ]] || { echo "  SKIP ${logical}/omlx/${spec} (omlx not on PATH)" >&2; continue; }
                run_cell "${logical}/omlx/${spec}"               "omlx"      "$mlxserve_path" "$spec" "" ""
                ;;
            "mlx-serve|alt")
                # mlx-serve loading the same .gguf LM Studio uses. PLD /
                # drafter silently no-op on the llama.cpp path (those are
                # MLX-only kernels), so the only meaningful spec here is
                # `none` — the SPECS list reflects that.
                if [[ -z "$mlxserve_gguf_path" || ! -e "$mlxserve_gguf_path" ]]; then
                    echo "  SKIP ${logical}/mlx-serve-gguf/${spec} (no mlxserve_gguf_path or file missing)" >&2
                    continue
                fi
                run_cell "${logical}/mlx-serve-gguf/${spec}"     "mlx-serve" "$mlxserve_gguf_path" "$spec" "" ""
                ;;
            "mlx-serve|")
                # Empty variant = MLX safetensors path (default).
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

# Render the engine-comparison chart only when there's something to compare —
# a bar chart of mlx-serve-only cells is just the CSV with extra steps.
if [[ "$INCLUDE_LMSTUDIO" -eq 1 || "$INCLUDE_OMLX" -eq 1 ]]; then
    mkdir -p "$(dirname "$PNG_OUT")"
    python3 "$SCRIPT_DIR/plot_vs_lmstudio_omlx.py" "$OUT" "$PNG_OUT" --family "$FAMILY"
    echo
    echo "=== chart written to $PNG_OUT ==="
else
    echo "(chart skipped — no comparison engines; pass --lmstudio/--omlx to render)" >&2
fi

# Optionally retain the raw CSV; otherwise it's a tempfile and gets cleaned
# up by the EXIT trap below.
if [[ -n "$KEEP_CSV" ]]; then
    mkdir -p "$(dirname "$KEEP_CSV")"
    cp "$OUT" "$KEEP_CSV"
    echo "CSV retained at $KEEP_CSV"
fi
