#!/bin/bash
# bench.sh — unified mlx-serve performance bench.
#
# Default run is mlx-serve only across {none,pld,drafter} × prefill/code
# prompts (fast dev-loop iteration; echo and free-form are opt-in via
# --echo / --freeform since 2026-07-14), MLX-format checkpoints only. Pass
# --lmstudio and/or --omlx to add the apples-to-apples comparison cells that
# produce charts in docs/perf-vs-lmstudio-omlx*.png — --lmstudio includes the
# LM Studio GGUF alt cell (the canonical chart's BASELINE); pass --gguf to
# also add the mlx-serve llama.cpp GGUF alt cell. Pass --concurrent N to also
# emit batched throughput rows (folded from the old bench_concurrent.py).
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
FREEFORM=0 # when 1, also measure the free-form/creative decode cell (retired
           # from the default matrix 2026-07-14 — it was a parity case on
           # every engine and cost a third of each cell's decode time; the
           # chart no longer plots it)
INCLUDE_ECHO=0 # when 1, also measure the echo (verbatim recitation) cell.
               # Retired from the default matrix 2026-07-14 alongside the
               # chart's echo panel: it is spec-decode's synthetic best case
               # (prompt-lookup's home turf) and doubled every cell's decode
               # time; code completion is the honest workload.
ISOLATE_SPECS=0 # when 1, boot a fresh mlx-serve per spec (the pre-2026-07-14
                # behavior). Default shares ONE boot per model across the
                # none/pld/drafter/mtp cells: every spec is already selected
                # per REQUEST via explicit enable_pld/enable_drafter/
                # enable_mtp body fields (build_body_mlx always sends all
                # three), so the per-spec launch flags were redundant — and
                # the shared boot cuts ~3 model loads per row.
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
INCLUDE_MTPLX=0
# --gguf gates ONLY the mlx-serve llama.cpp GGUF alt cell (opt-in — the
# mlx-serve side of the matrix is MLX-format by default). The LM Studio GGUF
# alt cell is NOT behind this flag: it rides --lmstudio, because it is the
# canonical chart's baseline (2026-07-14 — "what LM Studio users actually
# run" is the GGUF default, not the MLX beta path).
INCLUDE_GGUF=0
# --mlx-only: retained for back-compat; redundant unless --gguf is also
# passed (it then re-drops the mlx-serve GGUF alt cell).
MLX_ONLY=0
# MTPLX (github.com/youssofal/mtplx) — the reference native-MTP runtime.
# OpenAI-compatible `mtplx serve`. Its chat path REQUIRES an MTPLX-verified
# MTP artifact (a Youssofal/*-MTPLX-Optimized-* checkpoint or its target/
# +assistant/ pair) — plain mlx-community checkpoints fail at request time
# (its runtime can't consume our sidecar layout, and --stock-ar/--no-load-mtp
# still refuse with "MTP is not enabled"). So mtplx cells only make sense on
# rows whose checkpoint IS an MTPLX artifact; elsewhere they fail fast and
# the cell is skipped. Binary resolution: PATH first, then the local repo venv.
MTPLX_PORT=11252
MTPLX_BIN="${MTPLX_BIN:-}"
if [[ -z "$MTPLX_BIN" ]]; then
    if command -v mtplx >/dev/null 2>&1; then
        MTPLX_BIN="$(command -v mtplx)"
    elif [[ -x "$HOME/projects/agents/MTPLX/venv/bin/mtplx" ]]; then
        MTPLX_BIN="$HOME/projects/agents/MTPLX/venv/bin/mtplx"
    fi
fi
# Set by start_engine's mtplx arm from /v1/models (mtplx lowercases the dir
# basename into its served id — never guess it).
MTPLX_MODEL_ID=""
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
Usage: $0 --family <all|gemma|qwen36> [options]

The default run measures mlx-serve only across {none, pld, drafter, mtp} and
the prefill/code prompts (echo and free-form are opt-in). Add --lmstudio
and/or --omlx to include the comparison engines.

Options:
  --family NAME        Model matrix: 'all' (gemma + qwen36 combined — the
                       canonical single-chart run), 'gemma', or 'qwen36'
                       (required)
  --only SUBSTR        Only run targets whose logical name contains SUBSTR
  --specs SUBSTR       Only run spec entries containing SUBSTR (e.g.
                       'mlx-serve' or 'lmstudio') — split long rows across
                       invocations, then merge the --keep-csv slices
  --lmstudio           Include LM Studio cells (GGUF alt — the chart baseline
                       — + MLX where configured). Requires \`lms\` CLI; LM
                       Studio handles JIT model load on the warmup curl.
  --omlx               Include oMLX cells. Requires \`omlx\` on PATH; silently
                       skipped if missing.
  --mtplx              Include MTPLX cells (reference native-MTP runtime).
                       Only meaningful on rows whose checkpoint is an MTPLX
                       artifact (Youssofal/*-MTPLX-Optimized-*); other rows'
                       mtplx cells fail the warmup and are skipped.
  --gguf               Include the mlx-serve llama.cpp GGUF alt cell. Off by
                       default — the mlx-serve side of the matrix is
                       MLX-format only. (The LM Studio GGUF alt cell rides
                       --lmstudio, not this flag.)
  --mlx-only           Back-compat: re-drop the mlx-serve GGUF alt cell.
                       Redundant unless --gguf is passed.
  --concurrent N       Also emit a \`decode_c<N>\` row per mlx-serve cell. The
                       server is started with --max-concurrent N and N parallel
                       /v1/chat/completions are fired; the row's tok/s is the
                       aggregate rate. Default 0 (off).
  --out PATH           Chart PNG output path. Default is timestamped:
                       docs/perf-vs-lmstudio-omlx-<family>-YYYYMMDD-HHMMSS.png
                       The chart is skipped when no comparison engines are
                       enabled (a single-engine bar chart isn't useful).
  --keep-csv PATH      CSV output path. Default (since 2026-07-14):
                       docs/perf-csvs/bench-<family>-YYYYMMDD-HHMMSS.csv —
                       always retained (also on ctrl-C, with partial rows);
                       only a header-only run leaves nothing behind.
  --freeform           Also measure the free-form/creative decode cell
                       (dropped from the default matrix and chart 2026-07-14).
  --echo               Also measure the echo (verbatim recitation) cell
                       (dropped from the default matrix and chart 2026-07-14
                       — spec-decode's synthetic best case; code completion
                       is the honest default workload).
  --isolate-specs      Boot a fresh mlx-serve per spec cell instead of
                       sharing one boot per model (slower; the shared boot is
                       measurement-identical since specs are selected per
                       request).
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
        --freeform)   FREEFORM=1; shift ;;
        --echo)       INCLUDE_ECHO=1; shift ;;
        --isolate-specs) ISOLATE_SPECS=1; shift ;;
        --lmstudio)   INCLUDE_LMSTUDIO=1; shift ;;
        --omlx)       INCLUDE_OMLX=1; shift ;;
        --mtplx)      INCLUDE_MTPLX=1; shift ;;
        --gguf)       INCLUDE_GGUF=1; shift ;;
        --mlx-only)   MLX_ONLY=1; shift ;;
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

# The CSV is ALWAYS retained (2026-07-14): re-rendering / restyling a chart
# from a kept CSV is free, re-running the bench is an hour. --keep-csv PATH
# overrides the timestamped default. The run writes to a temp file first so
# an aborted run can't leave a half-written file at the final path; the EXIT
# trap copies whatever was measured (interrupts keep partial results — they
# merge into a later run's CSV via the --specs slicing workflow).
[[ -z "$KEEP_CSV" ]] && KEEP_CSV="docs/perf-csvs/bench-${FAMILY}-${TS}.csv"
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
declare -a TARGETS=()
# Shared path roots (used by both families' target definitions).
MD="$HOME/.mlx-serve/models"
DM="$MD/mlx-community"
LMS_DIR="$HOME/.lmstudio/models"
# Vendored GGUFs — same files LM Studio loads under the hood (the LM Studio
# GGUF alt cell rides --lmstudio; the .gguf FILE paths below feed only the
# opt-in --gguf mlx-serve llama.cpp cell).
GGUF_DIR="$LMS_DIR/lmstudio-community"

# LM Studio 0.3+ model IDs are the upstream HF org/name. Verify with
# `curl -sf http://127.0.0.1:1234/v1/models` — the MLX baseline lives under
# `mlx-community/<name>`, the GGUF alt under `google/<name>`. The script
# tolerates missing rows: any TARGETS entry whose `mlxserve_path` is absent
# skips silently. E2B was retired from the matrix 2026-07-13 (tiny-model row
# added wall time without informing engine comparisons).
add_gemma_targets() {
    TARGETS+=(
        "gemma4-e4b-4bit|$LMS_DIR/mlx-community/gemma-4-e4b-it-4bit|mlx-community/gemma-4-e4b-it|google/gemma-4-e4b|$DM/gemma-4-E4B-it-assistant-bf16|$GGUF_DIR/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
        "gemma4-12b-4bit|$LMS_DIR/mlx-community/gemma-4-12B-it-4bit|mlx-community/gemma-4-12b-it|google/gemma-4-12b|$DM/gemma-4-12B-it-assistant-bf16|$GGUF_DIR/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
        "gemma4-12b-qat-4bit|$LMS_DIR/mlx-community/gemma-4-12b-it-qat-4bit|mlx-community/gemma-4-12b-it-qat|google/gemma-4-12b-qat|$DM/gemma-4-12B-it-assistant-bf16|$GGUF_DIR/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"
        # 26B-A4B row: the QAT 4-bit checkpoint (the non-qat MLX dir was
        # retired). LM Studio advertises this side-loaded dir with a bare id
        # (no mlx-community/ prefix) — verified via /v1/models. NO drafter
        # dir (2026-07-14): the drafter cell on this MoE row measured the
        # known drafter-on-MoE regression — a config that defaults OFF in
        # the server — so it burned a boot + cells to chart a non-default
        # setup. PLD is the spec story on this row.
        "gemma4-26b-a4b-moe-qat-4bit|$LMS_DIR/mlx-community/gemma-4-26B-A4B-it-qat-4bit|gemma-4-26b-a4b-it-qat|google/gemma-4-26b-a4b||$GGUF_DIR/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"
        "gemma4-31b-4bit|$LMS_DIR/mlx-community/gemma-4-31b-it-4bit|mlx-community/gemma-4-31b-it|google/gemma-4-31b|$DM/gemma-4-31B-it-assistant-bf16|$GGUF_DIR/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
    )
}

# All engines load the same standard mlx-community 4-bit MLX weights (not
# the unsloth UD variants). The MTPLX-artifact row: 4-bit trunk +
# MTPLX-calibrated MTP adapter; mlx-serve loads it unmodified (mtp/ sidecar),
# MTPLX gets its native MTP — identical weights on every engine. It is the
# ONLY row where an mtplx cell can run.
#
# oMLX runs its own native MTP (mtp_enabled, see prepare_omlx_mtp) on the ONE
# qwen row whose checkpoint ships INLINE mtp weights (Qwen3.6-35B-A3B); on the
# 27B / mtplxopt rows the MTP head exists only as an mlx-serve `mtp/` sidecar,
# which omlx can't read, so its cell stays autoregressive there.
add_qwen36_targets() {
    # Qwen lms_alt ids are the FULL indexed identifiers
    # (publisher/repo/file.gguf): these GGUFs are side-loaded (no LM Studio
    # Hub virtual-model manifest, unlike the gemma `google/...` ids), so the
    # short hub id doesn't exist — but LM Studio JIT-loads the full
    # identifier fine (verified live 2026-07-14). The mtplxopt row has no
    # GGUF counterpart (MTPLX artifacts are MLX-only) — empty lms_alt.
    TARGETS+=(
        "qwen36-27b|$LMS_DIR/mlx-community/Qwen3.6-27B-4bit|qwen3.6-27b|lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf||$GGUF_DIR/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
        "qwen36-35b-a3b|$LMS_DIR/mlx-community/Qwen3.6-35B-A3B-4bit|qwen3.6-35b-a3b|lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf||$GGUF_DIR/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
        "qwen36-27b-mtplxopt|$LMS_DIR/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed|qwen3.6-27b-mtplx-optimized-speed|||"
    )
}

# Spec ordering rule (all families): mlx-serve MLX first (cool machine),
# mlx-serve GGUF next, omlx in the middle, LMS specs last so any thermal
# throttling that builds up during the row falls on the comparison engines,
# not on us. `mtp` cells only run on rows whose checkpoint ships an MTP
# sidecar (guarded in the dispatch loop); `drafter` cells only on rows with a
# drafter dir; `none`/`pld` pass --no-mtp so their labels stay truthful
# (priority is MTP > PLD when a sidecar is loaded).
case "$FAMILY" in
    gemma)
        add_gemma_targets
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::drafter" "mlx-serve:alt:none" "omlx:base:none" "mtplx:base:auto" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        # Gemma 4 has no thinking mode; the LMS workaround is a no-op.
        LMS_THINKING_WORKAROUND=0
        ;;
    qwen36)
        add_qwen36_targets
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::mtp" "mlx-serve:alt:none" "omlx:base:none" "mtplx:base:auto" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        # Qwen 3.6's chat template auto-activates `<think>` mode; LM Studio
        # ignores `chat_template_kwargs.enable_thinking:false`, so build_body_lms
        # uses the stacked workaround when this flag is on.
        LMS_THINKING_WORKAROUND=1
        ;;
    all)
        # The canonical combined matrix (one CSV → the single gemma+qwen
        # chart, tests/plot_vs_lmstudio_omlx.py --family all). Union of both
        # families' specs; per-row guards drop the inapplicable ones, and
        # LMS_THINKING_WORKAROUND is set PER ROW in the dispatch loop (Qwen
        # rows need it, Gemma rows must not carry the extra system message).
        add_gemma_targets
        add_qwen36_targets
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::drafter" "mlx-serve::mtp" "mlx-serve:alt:none" "omlx:base:none" "mtplx:base:auto" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        LMS_THINKING_WORKAROUND=0
        ;;
    *)
        echo "Unknown family '$FAMILY' (try all, gemma, or qwen36)" >&2
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

# --mlx-only: drop the mlx-serve GGUF alt cell — every remaining cell loads
# MLX-format safetensors.
if [[ "$MLX_ONLY" -eq 1 ]]; then
    _mlx_specs=()
    for _s in "${SPECS[@]}"; do
        [[ "$_s" == "mlx-serve:alt:"* ]] && continue
        _mlx_specs+=("$_s")
    done
    SPECS=("${_mlx_specs[@]}")
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

# MTPLX is OpenAI-compatible and honors chat_template_kwargs. The served model
# id is captured from /v1/models at boot (MTPLX_MODEL_ID) — it lowercases the
# checkpoint dir basename, so never derive it client-side.
build_body_mtplx() {
    local prompt="$1" mt="$2"
    local think=false; [[ "$THINKING" == "1" ]] && think=true
    jq -nc --arg p "$prompt" --arg model "$MTPLX_MODEL_ID" --argjson mt "$mt" --argjson think "$think" \
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
        mtplx)     port="$MTPLX_PORT" ;;
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
    case "$engine" in lmstudio) port="$LMS_PORT";; omlx) port="$OMLX_PORT";; mtplx) port="$MTPLX_PORT";; *) port="$SERVER_PORT";; esac
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
            mtplx)     body=$(build_body_mtplx "$(salted "$i" "$prompt")" "$mt") ;;
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
            mtplx)     body=$(build_body_mtplx "$(salted "$i" "$PREFILL_PROMPT")" 1) ;;
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
        mtplx)    port="$MTPLX_PORT" ;;
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
            mtplx)     body=$(build_body_mtplx "$(salted "c$i" "$prompt")" "$mt") ;;
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

# oMLX exposes native Qwen3.5/3.6 multi-token prediction as the per-model
# `mtp_enabled` flag in ~/.omlx/model_settings.json (keyed by model id = the
# served subdir basename), applied at engine construction. Turn it ON for the
# omlx cell so its Qwen bar is an apples-to-apples MTP comparison with
# mlx-serve's `mtp` cell — but only where it can actually engage:
#
#   - omlx's native MTP reads the head from INLINE `*.mtp.*` weights in the
#     checkpoint; it CANNOT use mlx-serve's `mtp/` sidecar layout. Among the
#     bench qwen rows only Qwen3.6-35B-A3B-4bit ships inline mtp tensors; the
#     27B / mtplxopt rows carry the head only as an mlx-serve sidecar, so omlx
#     stays autoregressive there.
#   - Forcing the flag on a checkpoint that declares `mtp_num_hidden_layers>0`
#     but ships no inline weights either no-ops (VLM load path skips the
#     attach) or FAILS strict weight-load (no-vision mlx-lm path). So gate the
#     flag on actual inline-weight presence rather than the config declaration.
#
# Gemma omlx cells run this too (native MTP is a Qwen feature) and harmlessly
# resolve to `mtp_enabled:false`. Idempotent read-modify-write that preserves
# any other persisted per-model settings; also self-heals a stale `true` left
# by an interactive omlx session on a sidecar-only checkpoint.
prepare_omlx_mtp() {
    local served_dir="$1" model_id="$2"
    python3 - "$served_dir" "$model_id" "$HOME/.omlx/model_settings.json" <<'PY'
import glob, json, os, struct, sys
served_dir, model_id, settings_path = sys.argv[1], sys.argv[2], sys.argv[3]

def has_inline_mtp(d):
    try:
        cfg = json.load(open(os.path.join(d, "config.json")))
    except Exception:
        return False
    tc = cfg.get("text_config", cfg)
    if int(tc.get("mtp_num_hidden_layers", 0) or 0) <= 0:
        return False
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
        except Exception:
            continue
        if any("mtp" in k.lower() for k in hdr if k != "__metadata__"):
            return True
    return False

enable = has_inline_mtp(served_dir)

data = {"version": 1, "models": {}}
if os.path.exists(settings_path):
    try:
        data = json.load(open(settings_path))
    except Exception:
        data = {"version": 1, "models": {}}
data.setdefault("version", 1)
data.setdefault("models", {}).setdefault(model_id, {})["mtp_enabled"] = bool(enable)

os.makedirs(os.path.dirname(settings_path), exist_ok=True)
tmp = settings_path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
os.replace(tmp, settings_path)
print(f"  omlx native MTP {'ENABLED' if enable else 'off'} for {model_id}",
      file=sys.stderr)
PY
}

stop_all_engines() {
    pkill -9 -x mlx-serve 2>/dev/null
    # oMLX launches as `python3 -m omlx.cli serve …` — match by `omlx.cli`.
    pkill -9 -f "omlx.cli" 2>/dev/null
    # MTPLX's server process carries the venv script path in its cmdline; the
    # port sweep below is the authoritative kill (a plain `pkill -f mtplx`
    # would be too broad).
    pkill -9 -f "bin/mtplx serve" 2>/dev/null
    # Belt-and-suspenders: clear known ports if anything survived.
    for p in "$SERVER_PORT" "$OMLX_PORT" "$MTPLX_PORT"; do
        local pids; pids="$(lsof -ti:"$p" 2>/dev/null)"
        [[ -n "$pids" ]] && echo "$pids" | xargs -r kill -9 2>/dev/null
    done
    # Wait for both ports to actually free up — pkill returns before the kernel
    # reaps the process, and a quick relaunch can race the old socket holding
    # the port (which then causes the new engine to die on bind). Plus give the
    # kernel time to reclaim the prior model's MLX buffers (5-20 GB) before the
    # next engine allocates — when this is too short, oMLX in particular has
    # been observed to die mid-prefill on the next cell.
    # Wait only for the ports we actually KILLED above (same list as the sweep).
    # NOT LMS_PORT: LM Studio's server is a persistent daemon we deliberately
    # never kill — its MODEL is freed by `lms unload --all`, not by dropping the
    # socket — so waiting on it can NEVER succeed and burns the full 30s timeout
    # + sleep 5 on EVERY start_engine. With --lmstudio that was ~40 s × 17 starts
    # = ~11 min of dead wait in a 20 min `--family all` run (measured 2026-07-16;
    # the run is ~4.7 min of actual generation). Waiting for a port you never
    # free is a category error — keep this list == the kill list above.
    local waited=0
    while (( waited < 30 )); do
        local busy=0
        for p in "$SERVER_PORT" "$OMLX_PORT" "$MTPLX_PORT"; do
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
                # Shared boot (default path): server defaults (--pld on, MTP
                # sidecar auto-load) + the drafter when the row has one. The
                # spec is selected per REQUEST — build_body_mlx always sends
                # explicit enable_pld/enable_drafter/enable_mtp — so this one
                # boot serves every mlx-serve cell of the row.
                multi)   [[ -n "$drafter" && -d "$drafter" ]] && extra="--drafter $drafter" ;;
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
            # Enable oMLX native MTP for this model where the checkpoint ships
            # inline mtp weights (Qwen3.6-35B-A3B); written before boot so the
            # ModelSettingsManager picks it up at engine construction.
            prepare_omlx_mtp "$model_or_path" "$model_id"
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
        mtplx)
            if [[ -z "$MTPLX_BIN" ]]; then
                echo "  mtplx binary not found (PATH or ~/projects/agents/MTPLX/venv/bin)" >&2
                return 1
            fi
            # `--generation-mode auto` is the only mode that serves chat: its
            # ar/--stock-ar/--no-load-mtp combinations all refuse requests
            # with "MTP is not enabled for this runtime" (probed live,
            # MTPLX 2.0.2). On a checkpoint without an MTPLX-compatible MTP
            # artifact the warmup below fails and the cell is skipped.
            # Thinking parity: MTPLX ignores chat_template_kwargs
            # .enable_thinking (probed: thinking_leaked=163 on the smoke
            # cell) — suppression is the server-side --reasoning flag.
            local reason_flag="off"
            [[ "$THINKING" == "1" ]] && reason_flag="on"
            "$MTPLX_BIN" serve --model "$model_or_path" --host 127.0.0.1 \
                --port "$MTPLX_PORT" --context-window "$CTX" \
                --generation-mode auto --reasoning "$reason_flag" --yes \
                >/tmp/bench_vs_lms_mtplx.log 2>&1 &
            local pid=$!
            ENGINE_PID="$pid"
            for i in $(seq 1 240); do
                curl -sf "http://127.0.0.1:$MTPLX_PORT/v1/models" >/dev/null 2>&1 && break
                sleep 0.5
                kill -0 "$pid" 2>/dev/null || { echo "  mtplx died (tail of /tmp/bench_vs_lms_mtplx.log:)" >&2; tail -n 10 /tmp/bench_vs_lms_mtplx.log >&2; return 1; }
                [[ "$i" -eq 240 ]] && { echo "  mtplx /v1/models never came up" >&2; return 1; }
            done
            # MTPLX lowercases the dir basename into its served id — read it
            # back instead of guessing.
            MTPLX_MODEL_ID="$(curl -sf "http://127.0.0.1:$MTPLX_PORT/v1/models" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d["data"][0]["id"])
except Exception:
    pass')"
            [[ -z "$MTPLX_MODEL_ID" ]] && { echo "  mtplx /v1/models gave no model id" >&2; return 1; }
            local warmup_body
            warmup_body=$(jq -nc --arg model "$MTPLX_MODEL_ID" '{model:$model,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false}')
            local warmup_resp
            warmup_resp=$(curl -sf -m 600 -X POST "http://127.0.0.1:$MTPLX_PORT/v1/chat/completions" \
                -H "Content-Type: application/json" -d "$warmup_body" 2>/dev/null)
            # MTPLX returns HTTP 200 with an {"error": …} body on runtime
            # incompatibility (the sidecar shape-mismatch class) — curl -f
            # can't see that, so check the body.
            if [[ -z "$warmup_resp" ]] || echo "$warmup_resp" | grep -q '"error"'; then
                echo "  mtplx warmup failed for $model_or_path — no MTPLX-compatible MTP artifact? (check /tmp/bench_vs_lms_mtplx.log)" >&2
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
    echo "  >> $label_prefix / $engine / $spec" >&2
    if ! start_engine "$engine" "$model_or_path" "$spec" "$drafter"; then
        echo "  SKIP $label_prefix" >&2
        return
    fi
    measure_cell "$label_prefix" "$engine" "$model_or_path" "$spec" "$drafter" "$notes" "$spec"
}

# Measure one cell against an ALREADY-BOOTED engine. `boot_spec` is what a
# mid-cell restart re-boots with: the cell's own spec on the isolated path,
# "multi" on the shared-boot path.
measure_cell() {
    local label_prefix="$1" engine="$2" model_or_path="$3" spec="$4" drafter="$5" notes="$6" boot_spec="$7"
    local label="${label_prefix}"

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
            if ! start_engine "$engine" "$model_or_path" "$boot_spec" "$drafter"; then
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

    # Default decode cell set is CODE ONLY (2026-07-14). Echo (--echo) is
    # spec-decode's synthetic best case and doubled every cell's decode time;
    # free-form (--freeform) measured parity on every engine. Code completion
    # is the honest workload the chart leads with.
    local kinds="code"
    [[ "$INCLUDE_ECHO" -eq 1 ]] && kinds="echo $kinds"
    [[ "$FREEFORM" -eq 1 ]] && kinds="decode $kinds"
    for kind in $kinds; do
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

# Shared-boot row runner (default mlx-serve path): ONE server boot per model
# serves every mlx-serve spec cell of the row — the spec is fully selected by
# the request body, so this is measurement-identical to per-spec boots while
# skipping ~3 redundant model loads per row. All cells share the same boot's
# thermal state, which is FAIRER within the row than staggered boots.
run_mlx_row() {
    local logical="$1" model_or_path="$2" drafter="$3"
    shift 3
    local specs=("$@")
    [[ ${#specs[@]} -eq 0 ]] && return
    echo "  >> ${logical}/mlx-serve/{${specs[*]}} (shared boot)" >&2
    if ! start_engine "mlx-serve" "$model_or_path" "multi" "$drafter"; then
        echo "  SKIP ${logical}/mlx-serve (shared boot failed)" >&2
        return
    fi
    local spec
    for spec in "${specs[@]}"; do
        measure_cell "${logical}/mlx-serve/${spec}" "mlx-serve" "$model_or_path" "$spec" "$drafter" "" "multi"
    done
}

cleanup() {
    stop_all_engines
    lms unload --all >/dev/null 2>&1 || true
    # Retain the CSV (runs on normal exit AND interrupts, so a ctrl-C'd run
    # keeps its partial rows). A header-only CSV (nothing measured — e.g. a
    # filtered dry run) is not worth littering docs/perf-csvs with.
    if [[ -n "$OUT" && -f "$OUT" ]]; then
        if [[ "$(wc -l < "$OUT")" -gt 1 ]]; then
            mkdir -p "$(dirname "$KEEP_CSV")"
            cp "$OUT" "$KEEP_CSV"
            echo "CSV retained at $KEEP_CSV"
        fi
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
HAS_MTPLX=0
if [[ "$INCLUDE_MTPLX" -eq 1 ]]; then
    if [[ -n "$MTPLX_BIN" && -x "$MTPLX_BIN" ]]; then
        HAS_MTPLX=1
    else
        echo "--mtplx passed but no mtplx binary found; cells will be skipped" >&2
    fi
fi
[[ "$INCLUDE_LMSTUDIO" -eq 0 && "$INCLUDE_OMLX" -eq 0 && "$INCLUDE_MTPLX" -eq 0 ]] && \
    echo "(mlx-serve only — pass --lmstudio/--omlx/--mtplx to add comparison engines)" >&2

for row in "${TARGETS[@]}"; do
    IFS='|' read -r logical mlxserve_path lms_baseline lms_alt drafter mlxserve_gguf_path <<<"$row"
    [[ -n "$ONLY" && "$logical" != *"$ONLY"* ]] && { echo "SKIP $logical (--only $ONLY)" >&2; continue; }
    [[ -d "$mlxserve_path" ]] || { echo "SKIP missing $mlxserve_path" >&2; continue; }
    # family=all mixes Gemma and Qwen rows; the LMS thinking workaround is a
    # Qwen-only need (its lighter combo adds a system message + /no_think
    # suffix that Gemma rows must not carry — prompt parity per row).
    if [[ "$FAMILY" == "all" ]]; then
        if [[ "$logical" == qwen* ]]; then LMS_THINKING_WORKAROUND=1; else LMS_THINKING_WORKAROUND=0; fi
    fi

    # ── Pass 1: mlx-serve MLX cells — collect the row's applicable specs and
    # run them against ONE shared boot (or per-spec boots with
    # --isolate-specs). mlx-serve entries lead every family's SPECS list, so
    # running them first preserves the cool-machine-first thermal ordering.
    row_mlx_specs=()
    for spec_entry in "${SPECS[@]}"; do
        [[ -n "$SPECS_ONLY" && "$spec_entry" != *"$SPECS_ONLY"* ]] && continue
        IFS=':' read -r engine variant spec <<<"$spec_entry"
        [[ "$engine|$variant" == "mlx-serve|" ]] || continue
        if [[ "$spec" == "drafter" && ( -z "$drafter" || ! -d "$drafter" ) ]]; then
            continue
        fi
        # The `mtp` cell only makes sense on rows whose checkpoint ships an
        # MTP sidecar (ANY of the four accepted layouts — mirrors
        # mtp.sidecar_rel_paths in src/mtp.zig, incl. the OptiQ layout) —
        # without one it would silently measure plain decode under an "mtp"
        # label. NOTE: this must be an any-of check per file — `ls a b c`
        # exits non-zero when ANY operand is missing, which silently
        # skipped the mtp cell on EVERY row in the 2026-07-14 run (no model
        # ships all three names).
        if [[ "$spec" == "mtp" &&
              ! -e "$mlxserve_path/mtp/weights.safetensors" &&
              ! -e "$mlxserve_path/mtp.safetensors" &&
              ! -e "$mlxserve_path/model-mtp.safetensors" &&
              ! -e "$mlxserve_path/optiq/mtp.safetensors" ]]; then
            continue
        fi
        row_mlx_specs+=("$spec")
    done
    if [[ ${#row_mlx_specs[@]} -gt 0 ]]; then
        if [[ "$ISOLATE_SPECS" -eq 1 ]]; then
            for spec in "${row_mlx_specs[@]}"; do
                run_cell "${logical}/mlx-serve/${spec}" "mlx-serve" "$mlxserve_path" "$spec" "$drafter" ""
            done
        else
            run_mlx_row "$logical" "$mlxserve_path" "$drafter" "${row_mlx_specs[@]}"
        fi
    fi

    # ── Pass 2: everything else (GGUF alt, comparison engines) — one boot
    # per cell as before. ──
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
                # The LM Studio GGUF cell is part of the default --lmstudio
                # set (2026-07-14): it is the canonical chart's BASELINE —
                # the llama.cpp GGUF path is what LM Studio users actually
                # run. --gguf gates only the mlx-serve llama.cpp alt below.
                [[ "$INCLUDE_LMSTUDIO" -eq 1 ]] || continue
                [[ -z "$lms_alt" ]] && continue
                run_cell "${logical}/lmstudio-alt/${spec}"       "lmstudio"  "$lms_alt"       "$spec" "" ""
                ;;
            "omlx|base")
                [[ "$INCLUDE_OMLX" -eq 1 ]] || continue
                [[ "$HAS_OMLX" -eq 1 ]] || { echo "  SKIP ${logical}/omlx/${spec} (omlx not on PATH)" >&2; continue; }
                run_cell "${logical}/omlx/${spec}"               "omlx"      "$mlxserve_path" "$spec" "" ""
                ;;
            "mtplx|base")
                [[ "$INCLUDE_MTPLX" -eq 1 ]] || continue
                [[ "$HAS_MTPLX" -eq 1 ]] || { echo "  SKIP ${logical}/mtplx/${spec} (mtplx binary not found)" >&2; continue; }
                run_cell "${logical}/mtplx/${spec}"              "mtplx"     "$mlxserve_path" "$spec" "" ""
                ;;
            "mlx-serve|alt")
                # mlx-serve loading the same .gguf LM Studio uses. Opt-in via
                # --gguf (the default matrix is MLX-format only). PLD /
                # drafter silently no-op on the llama.cpp path (those are
                # MLX-only kernels), so the only meaningful spec here is
                # `none` — the SPECS list reflects that.
                [[ "$INCLUDE_GGUF" -eq 1 ]] || continue
                if [[ -z "$mlxserve_gguf_path" || ! -e "$mlxserve_gguf_path" ]]; then
                    echo "  SKIP ${logical}/mlx-serve-gguf/${spec} (no mlxserve_gguf_path or file missing)" >&2
                    continue
                fi
                run_cell "${logical}/mlx-serve-gguf/${spec}"     "mlx-serve" "$mlxserve_gguf_path" "$spec" "" ""
                ;;
            "mlx-serve|")
                # Handled in pass 1 (shared-boot path).
                continue
                ;;
        esac
    done
done

stop_all_engines
lms unload --all >/dev/null 2>&1 || true

# Render the engine-comparison chart only when there's something to compare —
# a bar chart of mlx-serve-only cells is just the CSV with extra steps.
if [[ "$INCLUDE_LMSTUDIO" -eq 1 || "$INCLUDE_OMLX" -eq 1 || "$INCLUDE_MTPLX" -eq 1 ]]; then
    mkdir -p "$(dirname "$PNG_OUT")"
    python3 "$SCRIPT_DIR/plot_vs_lmstudio_omlx.py" "$OUT" "$PNG_OUT" --family "$FAMILY"
    echo
    echo "=== chart written to $PNG_OUT ==="
else
    echo "(chart skipped — no comparison engines; pass --lmstudio/--omlx to render)" >&2
fi

# CSV retention happens in the EXIT trap (cleanup) so interrupted runs keep
# their partial rows too.
