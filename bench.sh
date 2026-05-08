#!/bin/bash
# bench.sh — top-level benchmark helper.
#
# Templated wrapper around bench_run.sh. Each template hardcodes a model set
# and the spec-decode configurations it makes sense to test on those models.
# Engines are kept apples-to-apples via OpenAI HTTP chat completions with
# matched params (temp 0, greedy, thinking off, fixed ctx).
#
# Replaces the legacy bench.sh + tests/bench_spec.sh + tests/bench_spec_matrix.sh.
#
# See `./bench.sh --help` for usage.
set -uo pipefail

CORE="$(dirname "$0")/bench_run.sh"
MODELS_DIR="$HOME/.mlx-serve/models"

# ── Logical-name → per-engine identifier table ──
# Pipe-separated: logical|mlx-serve-path|mlx-lm-path|lmstudio-key|drafter-dir(or empty)
MODEL_TABLE=(
    # MoE
    "gemma26b-moe|$MODELS_DIR/gemma-4-26b-a4b-it-4bit|$MODELS_DIR/gemma-4-26b-a4b-it-4bit|gemma-4-26b-a4b-it|$MODELS_DIR/mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
    "qwen35b-moe|$MODELS_DIR/Qwen3.6-35B-A3B-6bit|$MODELS_DIR/Qwen3.6-35B-A3B-6bit|qwen3.6-35b-a3b@6bit|"
    # MoE + native MTP head (Qwen 3.6 35B-A3B with the MTP weights baked in)
    "qwen35b-moe-mtp|$MODELS_DIR/mlx-community/Qwen3.6-35B-A3B-mtp-4bit|$MODELS_DIR/mlx-community/Qwen3.6-35B-A3B-mtp-4bit|||"
    # Dense
    "gemma31b|$MODELS_DIR/gemma-4-31b-it-8bit|$MODELS_DIR/gemma-4-31b-it-8bit|gemma-4-31b-it|$MODELS_DIR/mlx-community/gemma-4-31B-it-assistant-bf16"
    "qwen36-27b|$MODELS_DIR/Qwen3.6-27B-4bit|$MODELS_DIR/Qwen3.6-27B-4bit|qwen/qwen3.6-27b|"
    # Dense + native MTP head (Qwen 3.6 27B with MTP weights)
    "qwen36-27b-mtp-v2|$MODELS_DIR/mlx-community/Qwen3.6-27B-mtp|$MODELS_DIR/mlx-community/Qwen3.6-27B-mtp|||"
    # Drafter pairings (Gemma 4 dense + assistant). Drafter dir is the small
    # 4-layer assistant Google ships alongside Gemma 4 — only mlx-serve
    # supports it, and the bench skips the drafter spec cell if the dir is
    # absent on disk.
    "gemma4-e4b|$MODELS_DIR/gemma-4-e4b-it-8bit|$MODELS_DIR/gemma-4-e4b-it-8bit|gemma-4-e4b-it|$MODELS_DIR/mlx-community/gemma-4-E4B-it-assistant-bf16"
    # MTP (Qwen MTPLX speed builds — full-model MTP fork)
    "qwen35-4b-mtp|$MODELS_DIR/Qwen3.5-4B-MTPLX-Optimized-Speed|$MODELS_DIR/Qwen3.5-4B-MTPLX-Optimized-Speed|||"
    "qwen36-27b-mtp|$MODELS_DIR/Qwen3.6-27B-MTPLX-Optimized-Speed|$MODELS_DIR/Qwen3.6-27B-MTPLX-Optimized-Speed|||"
)

# Templates: name → "logical_models|specs_for_mlxserve"
# Spec sets: `none,pld` for moe/dense; `none,drafter` for drafter; `none,mtp` for mtp.
# When --engine all is used, non-mlx-serve engines run the `none` baseline only
# (since they don't support drafter/mtp, and even pld is mlx-serve-specific).
TEMPLATE_MOE_MODELS=("gemma26b-moe" "qwen35b-moe")
TEMPLATE_MOE_SPECS=("none" "pld")

TEMPLATE_DENSE_MODELS=("gemma31b" "qwen36-27b")
TEMPLATE_DENSE_SPECS=("none" "pld")

TEMPLATE_DRAFTER_MODELS=("gemma4-e4b" "gemma26b-moe" "gemma31b")
TEMPLATE_DRAFTER_SPECS=("none" "drafter")

TEMPLATE_MTP_MODELS=("qwen35-4b-mtp" "qwen36-27b-mtp" "qwen36-27b-mtp-v2" "qwen35b-moe-mtp")
TEMPLATE_MTP_SPECS=("none" "mtp")

# ── Default args ──
TEMPLATE=""
ENGINE="mlx-serve"
RUNS=3
MAX_TOKENS=192
CTX_SIZE=4096
BINARY="./zig-out/bin/mlx-serve"
CORPUS=0
ECHO=0

usage() {
    cat <<EOF
Usage: bench.sh <template> [options]

Templates (hardcoded model + spec sets):
  moe         Gemma 4 26B-A4B  +  Qwen 3.6 35B-A3B          [specs: none, pld]
  dense       Gemma 4 31B      +  Qwen 3.6 27B              [specs: none, pld]
  drafter     Gemma 4 E4B + assistant drafter               [specs: none, drafter]
              (drafter spec runs on mlx-serve only)
  mtp         Qwen 3.5 4B + Qwen 3.6 27B (MTPLX builds)     [specs: none, mtp]
              (mtp spec runs on mlx-serve only)
  --corpus    PLD threshold-tuning corpus on Gemma 4 E4B.
              9 prompts × pld-on/pld-off; reports per-prompt n-gram score and
              ratio. (No --engine; mlx-serve only.)

Options:
  --engine <name>   mlx-serve | mlx-lm | lmstudio | all     (default: mlx-serve)
  --runs <n>        Repeats per cell; run 1 is warmup       (default: 3, min: 2)
  --max-tokens <n>  Decode budget                           (default: 192)
  --ctx-size <n>    Context size hint to the engine         (default: 4096)
  --binary <path>   mlx-serve binary                        (default: ./zig-out/bin/mlx-serve)
  --echo            Add an extra heavy-echo decode cell to each run.
                    Recitation-style prompt that exercises spec-decode
                    n-gram lookahead — the workload PLD/MTP/drafter target.
  -h, --help        This message

Output: pipe-separated rows on stdout.
  label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|notes

Examples:
  bench.sh moe --engine all
  bench.sh dense --engine mlx-serve --runs 5
  bench.sh drafter
  bench.sh --corpus
EOF
}

[[ $# -eq 0 ]] && { usage; exit 0; }

# Two-pass arg parse: first positional = template (unless --corpus is given).
while [[ $# -gt 0 ]]; do
    case "$1" in
        moe|dense|drafter|mtp)  TEMPLATE="$1"; shift ;;
        --corpus)               CORPUS=1; shift ;;
        --engine)               ENGINE="$2"; shift 2 ;;
        --runs)                 RUNS="$2"; shift 2 ;;
        --max-tokens)           MAX_TOKENS="$2"; shift 2 ;;
        --ctx-size)             CTX_SIZE="$2"; shift 2 ;;
        --binary)               BINARY="$2"; shift 2 ;;
        --echo)                 ECHO=1; shift ;;
        -h|--help)              usage; exit 0 ;;
        *)                      echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$CORPUS" == "0" && -z "$TEMPLATE" ]]; then
    echo "Error: pass a template (moe|dense|drafter|mtp) or --corpus" >&2
    echo
    usage
    exit 1
fi

# ── Logical-name lookup ──
# resolve_model <logical> <engine> → echoes "model_id|drafter_dir" (drafter_dir empty if N/A)
# Returns nonzero if no entry for that engine.
resolve_model() {
    local logical="$1" engine="$2"
    for row in "${MODEL_TABLE[@]}"; do
        IFS='|' read -r ln mlxserve mlxlm lmstudio drafter <<<"$row"
        [[ "$ln" != "$logical" ]] && continue
        case "$engine" in
            mlx-serve) [[ -z "$mlxserve" ]] && return 1; echo "$mlxserve|$drafter" ;;
            mlx-lm)    [[ -z "$mlxlm"    ]] && return 1; echo "$mlxlm|$drafter" ;;
            lmstudio)  [[ -z "$lmstudio" ]] && return 1; echo "$lmstudio|$drafter" ;;
            *)         return 1 ;;
        esac
        return 0
    done
    return 1
}

# Engines list to run for this invocation.
engines_for() {
    if [[ "$ENGINE" == "all" ]]; then
        echo "mlx-serve mlx-lm lmstudio"
    else
        echo "$ENGINE"
    fi
}

# Spec applies to engine? Drafter/mtp are mlx-serve only; pld is mlx-serve only too
# (but other engines run the baseline for comparability via spec=none). Returns 0
# if the (engine, spec) combo is supported.
spec_applies() {
    local engine="$1" spec="$2"
    case "$spec" in
        none) return 0 ;;
        pld|mtp|drafter) [[ "$engine" == "mlx-serve" ]] && return 0 || return 1 ;;
        *) return 1 ;;
    esac
}

run_cell() {
    local logical="$1" engine="$2" spec="$3"
    local resolved
    if ! resolved=$(resolve_model "$logical" "$engine"); then
        echo "  SKIP: $engine has no entry for $logical" >&2
        return
    fi
    local model="${resolved%%|*}"
    local drafter="${resolved#*|}"

    if ! spec_applies "$engine" "$spec"; then
        echo "  SKIP: $engine does not support spec=$spec for $logical" >&2
        return
    fi

    # Sanity-check local paths exist (lmstudio is by key — skip path check).
    if [[ "$engine" != "lmstudio" && ! -d "$model" ]]; then
        echo "  SKIP: $model not found on disk ($logical/$engine)" >&2
        return
    fi
    if [[ "$spec" == "drafter" && ( -z "$drafter" || ! -d "$drafter" ) ]]; then
        echo "  SKIP: drafter dir missing for $logical" >&2
        return
    fi

    local args=(
        --engine "$engine"
        --model "$model"
        --spec "$spec"
        --runs "$RUNS"
        --max-tokens "$MAX_TOKENS"
        --ctx-size "$CTX_SIZE"
        --binary "$BINARY"
        --label "${logical}/${engine}/${spec}"
    )
    [[ -n "$drafter" && "$spec" == "drafter" ]] && args+=(--drafter-dir "$drafter")
    [[ "$ECHO" == "1" ]] && args+=(--echo)

    "$CORE" "${args[@]}"
}

# ── Corpus mode: PLD threshold tuning ──
# 9 prompts × Gemma 4 E4B × (pld-off baseline + pld-on gated). Prints per-prompt
# n-gram score, gate decision, baseline tps, pld tps, ratio. Server log is
# parsed for the gate's decision line.
run_corpus() {
    local model="$MODELS_DIR/gemma-4-e4b-it-4bit"
    [[ -d "$model" ]] || { echo "ERROR: corpus mode needs Gemma 4 E4B at $model" >&2; exit 1; }
    local port=11241
    pkill -f mlx-serve 2>/dev/null; sleep 0.5
    "$BINARY" --model "$model" --serve --port "$port" --pld --log-level info \
        >/tmp/bench_corpus.log 2>&1 &
    local pid=$!
    trap "kill $pid 2>/dev/null; wait $pid 2>/dev/null" EXIT
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
        sleep 0.5
    done

    local labels=(heavy-echo code-rename json-transform rag-qa agent-turn plain-qa code-translate summarize creative)
    local prompts=(
'Repeat the following Python code verbatim, but rename the function `compute_total` to `total`:
```python
def compute_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    if total > 100:
        total *= 0.9
    return total
```
Output ONLY the renamed code, nothing else.'
'Rename `getUserById` to `findUser` in this code, output only the renamed code:
function getUserById(id) {
    const user = database.users.find(u => u.id === id);
    if (!user) return null;
    return user;
}'
'Convert this list of users to a JSON array, one object per user with keys "name" and "age":
- Alice, 30
- Bob, 25
- Charlie, 42
- Dana, 18
Output only the JSON, no explanation.'
'Context: Apple announced the M5 chip in October 2026. The chip features 12 performance cores and 8 efficiency cores, fabricated on a 2nm process. It has a unified memory bandwidth of 800 GB/s and supports up to 256GB of LPDDR6 memory.

Question: How much memory bandwidth does the M5 chip have? Answer in one short sentence.'
'You are a coding agent. The user said: "fix the typo in src/auth.zig". Reply with a one-line plan: which tool you would call first and what arguments you would pass to it.'
'What is the capital of Australia? Answer in one sentence.'
'Translate this Python function to TypeScript. Output only the TypeScript:
def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]'
'Summarize the following text in exactly two sentences:

The 2026 Antarctic Treaty conference reaffirmed the moratorium on commercial mineral extraction in the polar region. Delegates from 54 nations agreed to extend the treaty for an additional 50 years, citing accelerating ice loss measurements gathered by the joint US-EU SCOTT-II satellite array launched the prior year. Several signatories pushed for a stricter biosphere protection clause, but consensus stopped short of a binding limit on tourist vessel traffic, which has tripled since 2020.'
'Write a 30-line poem about a lighthouse keeper at the end of the world. Use vivid imagery.'
    )

    echo "label|prompt|ngram_score|gate_decision|baseline_tps|pld_tps|ratio"
    for i in "${!labels[@]}"; do
        local label="${labels[$i]}"
        local prompt="${prompts[$i]}"
        # baseline: pld off
        : > /tmp/bench_corpus.log
        local body=$(jq -nc --arg p "$prompt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false,enable_pld:false}')
        curl -sf -m 90 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" -d "$body" -o /dev/null
        local btps=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' /tmp/bench_corpus.log | tail -1 | grep -aoE '[0-9]+\.[0-9]+' | head -1)
        # gated: pld on, gate decides
        : > /tmp/bench_corpus.log
        body=$(jq -nc --arg p "$prompt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false}')
        curl -sf -m 90 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" -d "$body" -o /dev/null
        local ptps=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' /tmp/bench_corpus.log | tail -1 | grep -aoE '[0-9]+\.[0-9]+' | head -1)
        local ngram=$(LC_ALL=C grep -aoE 'spec-gate: ngram-score=[0-9.]+' /tmp/bench_corpus.log | tail -1 | grep -aoE '[0-9]+\.[0-9]+' | head -1)
        local gate=$(LC_ALL=C grep -aq 'pld=disabled (ngram-score' /tmp/bench_corpus.log && echo "disabled" || echo "enabled")
        btps=${btps:-0}; ptps=${ptps:-0}; ngram=${ngram:-0}
        local ratio=$(python3 -c "p=float('$ptps'); b=float('$btps'); print(f'{p/b:.2f}' if b>0 else '0')")
        echo "corpus|$label|$ngram|$gate|$btps|$ptps|$ratio"
    done
}

# ── Dispatch ──
echo "label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|notes"

if [[ "$CORPUS" == "1" ]]; then
    run_corpus
    exit 0
fi

case "$TEMPLATE" in
    moe)     models=("${TEMPLATE_MOE_MODELS[@]}");     specs=("${TEMPLATE_MOE_SPECS[@]}") ;;
    dense)   models=("${TEMPLATE_DENSE_MODELS[@]}");   specs=("${TEMPLATE_DENSE_SPECS[@]}") ;;
    drafter) models=("${TEMPLATE_DRAFTER_MODELS[@]}"); specs=("${TEMPLATE_DRAFTER_SPECS[@]}") ;;
    mtp)     models=("${TEMPLATE_MTP_MODELS[@]}");     specs=("${TEMPLATE_MTP_SPECS[@]}") ;;
    *)       echo "unreachable"; exit 1 ;;
esac

for engine in $(engines_for); do
    for logical in "${models[@]}"; do
        for spec in "${specs[@]}"; do
            run_cell "$logical" "$engine" "$spec"
        done
    done
done
