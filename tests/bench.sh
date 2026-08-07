#!/bin/bash
# bench.sh — the performance bench. llmprobe measures; this drives the engines.
#
# One `llmprobe --bench-only` run per engine per model produces BOTH the
# headline decode/prefill/TTFT medians and the context-scaling ladder, so a
# release takes one run and one CSV where it used to take two protocols
# (bench.sh's own curl timing + mtp_ladder_pair.sh) and two files.
#
# Why llmprobe rather than hand-rolled cells: it discards a warmup per
# scenario, reports median-of-3 as `median (min-max)`, refuses to fabricate a
# number when usage is missing, records the machine it ran on, and applies the
# same protocol to every engine — none of which our curl loops did.
#
#   ./tests/bench.sh                                  # mlx-serve only, all models
#   ./tests/bench.sh --only qwen36-27b                # one model row
#   ./tests/bench.sh --lmstudio --omlx --mtplx --full # the release/marketing run
#
# What each cell is: the engine booted with its SHIPPING DEFAULTS. mlx-serve
# picks its own speculative mode (MTP > PLD) per checkpoint, so the bar is what
# a user gets out of the box; llmprobe's speculative probe reports what that
# turned out to be (ratio + tokens/step land in the CSV).
#
# Output (all from ONE run):
#   docs/perf-csvs/probe-<tag>.csv                    — every cell + every rung
#   docs/perf-pngs/perf-vs-lmstudio-omlx-all-<tag>.png — headline bar chart
#   docs/perf-pngs/perf-mtp-ladder-<tag>.png           — context ladder
#
# Requirements: node (npx), python3 + matplotlib, curl. mlx-serve built
# ReleaseFast (Debug is 2-4x slower = a fake regression). Comparison engines
# are skipped with a note when their binary is absent, never an error.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
FAMILY="all"
ONLY=""
FULL=0            # --full: ladder climbs to 32k/64k and takes median-of-3 per rung
INCLUDE_LMSTUDIO=0
INCLUDE_OMLX=0
INCLUDE_MTPLX=0
LADDER_MODEL="qwen36-27b"
TAG="$(date +%Y%m%d-%H%M%S)"
CSV_OUT=""
PNG_BAR=""
PNG_LADDER=""
JSON_DIR=""
NO_PLOT=0
SETTLE="${SETTLE:-20}"

BINARY="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
MTPLX_BIN="${MTPLX_BIN:-$HOME/projects/agents/MTPLX/.venv/bin/mtplx}"
[ -x "$MTPLX_BIN" ] || MTPLX_BIN="$HOME/.mtplx/bin/mtplx"
OMLX_BIN="${OMLX_BIN:-/Applications/oMLX.app/Contents/MacOS/omlx-cli}"
LLMPROBE="${LLMPROBE:-npx -y llmprobe@latest}"

SERVER_PORT=11250
OMLX_PORT=11251
MTPLX_PORT=11252
LMS_PORT=1234

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)       FAMILY="$2"; shift 2 ;;
        --only)         ONLY="$2"; shift 2 ;;
        --full)         FULL=1; shift ;;
        --lmstudio)     INCLUDE_LMSTUDIO=1; shift ;;
        --omlx)         INCLUDE_OMLX=1; shift ;;
        --mtplx)        INCLUDE_MTPLX=1; shift ;;
        --ladder-model) LADDER_MODEL="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --csv)          CSV_OUT="$2"; shift 2 ;;
        --png-bar)      PNG_BAR="$2"; shift 2 ;;
        --png-ladder)   PNG_LADDER="$2"; shift 2 ;;
        --json-dir)     JSON_DIR="$2"; shift 2 ;;
        --no-plot)      NO_PLOT=1; shift ;;
        --settle)       SETTLE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown flag: $1 (try --help)" >&2; exit 1 ;;
    esac
done

CSV_OUT="${CSV_OUT:-$ROOT/docs/perf-csvs/probe-$TAG.csv}"
PNG_BAR="${PNG_BAR:-$ROOT/docs/perf-pngs/perf-vs-lmstudio-omlx-all-$TAG.png}"
PNG_LADDER="${PNG_LADDER:-$ROOT/docs/perf-pngs/perf-mtp-ladder-$TAG.png}"
# Reports live outside the repo by default: they are the raw material, the CSV
# is the artifact. ~/claude-tmp survives reboots; /tmp does not.
JSON_DIR="${JSON_DIR:-$HOME/claude-tmp/bench-$TAG}"
mkdir -p "$JSON_DIR"

# ── Model matrix ──
# logical|mlxserve_path|lms_mlx_id|lms_gguf_id
# An empty LM Studio id skips that cell for the row; a missing mlxserve_path
# skips the whole row (silently — a bench you can't run on this box shouldn't
# be an error on the box that can).
MD="$HOME/.mlx-serve/models"
DM="$MD/mlx-community"
LMS_DIR="$HOME/.lmstudio/models"

declare -a TARGETS=()
add_gemma_targets() {
    TARGETS+=(
        "gemma4-e4b-4bit|$LMS_DIR/mlx-community/gemma-4-e4b-it-4bit|mlx-community/gemma-4-e4b-it|google/gemma-4-e4b"
        "gemma4-26b-a4b-moe-qat-4bit|$LMS_DIR/mlx-community/gemma-4-26B-A4B-it-qat-4bit|gemma-4-26b-a4b-it-qat|google/gemma-4-26b-a4b"
        "gemma4-31b-4bit|$LMS_DIR/mlx-community/gemma-4-31b-it-4bit|mlx-community/gemma-4-31b-it|google/gemma-4-31b"
    )
}
# qwen36-27b is oMLX's OWN oQ4e checkpoint with inline MTP weights, loaded
# unmodified by mlx-serve too — same weights, both native MTP paths. The LM
# Studio ids stay the vanilla MLX/GGUF repos (LM Studio can't load the
# inline-mtp layout), so those bars are the plain-AR reference.
# qwen36-27b-mtplxopt is the ONLY row an MTPLX cell may run on: MTPLX emits
# garbage on foreign checkpoints, and a decode rate over garbage is not a
# comparison. It has no GGUF counterpart.
add_qwen36_targets() {
    TARGETS+=(
        "qwen36-27b|$MD/Jundot/Qwen3.6-27B-oQ4e-mtp|qwen3.6-27b|lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
        "qwen36-35b-a3b|$MD/stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP|qwen3.6-35b-a3b|lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
        "qwen36-27b-mtplxopt|$LMS_DIR/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed|qwen3.6-27b-mtplx-optimized-speed|"
    )
}

case "$FAMILY" in
    gemma)  add_gemma_targets ;;
    qwen36) add_qwen36_targets ;;
    all)    add_gemma_targets; add_qwen36_targets ;;
    *) echo "Unknown family '$FAMILY' (try all, gemma, qwen36)" >&2; exit 1 ;;
esac

# ── Engine lifecycle ──
# RULE (cost 11 min/run when broken): the port wait-list must equal the kill
# list. LM Studio's server is a persistent daemon we deliberately never kill —
# its MODEL is freed by `lms unload --all`, not by dropping the socket — so
# LMS_PORT must never appear here or every stop burns the full timeout.
stop_all_engines() {
    pkill -f "mlx-serve --serve" 2>/dev/null
    pkill -f "mtplx serve" 2>/dev/null
    pkill -f omlx-cli 2>/dev/null
    pkill -f omlx-server 2>/dev/null
    for _ in $(seq 1 30); do
        local busy=0
        for port in "$SERVER_PORT" "$OMLX_PORT" "$MTPLX_PORT"; do
            lsof -ti tcp:"$port" >/dev/null 2>&1 && busy=1
        done
        [[ "$busy" -eq 0 ]] && return 0
        sleep 1
    done
}
trap 'stop_all_engines' EXIT

wait_url() { # url tries
    for _ in $(seq 1 "$2"); do
        curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# JIT-load warmup doubling as a load-completed probe: LM Studio and oMLX load
# on first request; the others load at boot and pass straight through.
warmup() { # port model_id
    curl -sf -m 900 "http://127.0.0.1:$1/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
        >/dev/null 2>&1
}

# oMLX rejects unauthenticated requests unless this is flipped; the bench sends
# no Authorization header.
prepare_omlx() {
    local settings="$HOME/.omlx/settings.json"
    [[ -f "$settings" ]] || return 0
    python3 - "$settings" <<'PY' 2>/dev/null || true
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
auth = cfg.setdefault("auth", {})
if auth.get("skip_api_key_verification") is not True:
    auth["skip_api_key_verification"] = True
    json.dump(cfg, open(path, "w"), indent=2)
PY
}

probe() { # model engine spec port model_id
    CELL_RAN=1
    local out="$JSON_DIR/$1__$2__$3.json"
    local depth=(--bench-only)
    [[ "$FULL" -eq 1 ]] && depth+=(--full)
    echo "── $1 / $2 (:$4, $5) ──"
    # shellcheck disable=SC2086
    $LLMPROBE "localhost:$4" -m "$5" "${depth[@]}" --save "$out" \
        || echo "  llmprobe failed for $1/$2" >&2
}

# ── Cells ──
# Ordering rule: mlx-serve FIRST on every row, so any thermal drift that builds
# during the row lands on the comparison engines rather than on us. llmprobe's
# own sustained-load check reports drift within a cell; this handles across.
cell_mlx_serve() { # logical path
    [[ -x "$BINARY" ]] || { echo "skip $1/mlx-serve: no $BINARY" >&2; return; }
    "$BINARY" --serve --model "$2" --port "$SERVER_PORT" \
        >"$JSON_DIR/$1__mlx-serve.log" 2>&1 &
    local pid=$!
    if wait_url "http://127.0.0.1:$SERVER_PORT/health" 300; then
        probe "$1" "mlx-serve" "default" "$SERVER_PORT" "$(basename "$2")"
    else
        echo "  mlx-serve never came up for $1" >&2
    fi
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}

cell_omlx() { # logical path
    [[ "$INCLUDE_OMLX" -eq 1 ]] || return
    [[ -x "$OMLX_BIN" ]] || { echo "skip $1/omlx: no $OMLX_BIN" >&2; return; }
    prepare_omlx
    "$OMLX_BIN" serve --model-dir "$(dirname "$2")" --port "$OMLX_PORT" \
        >"$JSON_DIR/$1__omlx.log" 2>&1 &
    local pid=$!
    if wait_url "http://127.0.0.1:$OMLX_PORT/v1/models" 300; then
        warmup "$OMLX_PORT" "$(basename "$2")"
        probe "$1" "omlx" "default" "$OMLX_PORT" "$(basename "$2")"
    else
        echo "  omlx never came up for $1" >&2
    fi
    kill "$pid" 2>/dev/null
    pkill -f omlx-cli 2>/dev/null; pkill -f omlx-server 2>/dev/null
}

cell_mtplx() { # logical path
    [[ "$INCLUDE_MTPLX" -eq 1 ]] || return
    [[ -x "$MTPLX_BIN" ]] || { echo "skip $1/mtplx: no mtplx binary" >&2; return; }
    # MTPLX's MTP path is calibrated per artifact: on a foreign checkpoint it
    # emits gibberish while still reporting a decode rate, and charting that
    # rate would be measuring garbage output rather than an engine.
    if [[ "$(basename "$2")" != *MTPLX-Optimized* ]]; then
        echo "skip $1/mtplx: not an MTPLX-calibrated artifact" >&2; return
    fi
    "$MTPLX_BIN" serve --model "$2" --port "$MTPLX_PORT" \
        >"$JSON_DIR/$1__mtplx.log" 2>&1 &
    local pid=$!
    if wait_url "http://127.0.0.1:$MTPLX_PORT/health" 300; then
        local id
        id="$(curl -sf "http://127.0.0.1:$MTPLX_PORT/health" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["model"])' 2>/dev/null \
            || basename "$2")"
        probe "$1" "mtplx" "default" "$MTPLX_PORT" "$id"
    else
        echo "  mtplx never came up for $1" >&2
    fi
    kill "$pid" 2>/dev/null; pkill -f "mtplx serve" 2>/dev/null
}

cell_lmstudio() { # logical variant model_id
    [[ "$INCLUDE_LMSTUDIO" -eq 1 ]] || return
    [[ -n "$3" ]] || return
    command -v lms >/dev/null || { echo "skip $1/$2: no lms CLI" >&2; return; }
    lms server start --port "$LMS_PORT" >/dev/null 2>&1
    wait_url "http://127.0.0.1:$LMS_PORT/v1/models" 60 || {
        echo "  lmstudio never came up" >&2; return; }
    lms unload --all >/dev/null 2>&1
    # `lms load` silently hangs on some releases; HTTP JIT-load is the
    # supported path.
    warmup "$LMS_PORT" "$3" || { echo "  lmstudio JIT-load failed for $3" >&2; return; }
    probe "$1" "$2" "default" "$LMS_PORT" "$3"
    lms unload --all >/dev/null 2>&1
}

# ── Run ──
echo "=== bench: family=$FAMILY depth=$([[ $FULL -eq 1 ]] && echo full || echo default) tag=$TAG ==="
echo "    reports → $JSON_DIR"
stop_all_engines

for row in "${TARGETS[@]}"; do
    IFS='|' read -r logical mlxserve_path lms_mlx_id lms_gguf_id <<< "$row"
    [[ -n "$ONLY" && "$logical" != *"$ONLY"* ]] && continue
    if [[ ! -e "$mlxserve_path" ]]; then
        echo "SKIP $logical (no checkpoint at $mlxserve_path)" >&2
        continue
    fi
    echo
    echo ">> $logical"
    for engine_cell in mlx-serve omlx mtplx lms-mlx lms-gguf; do
        CELL_RAN=0
        case "$engine_cell" in
            mlx-serve) cell_mlx_serve "$logical" "$mlxserve_path" ;;
            omlx)      cell_omlx      "$logical" "$mlxserve_path" ;;
            mtplx)     cell_mtplx     "$logical" "$mlxserve_path" ;;
            lms-mlx)   cell_lmstudio  "$logical" "lmstudio-baseline" "$lms_mlx_id" ;;
            lms-gguf)  cell_lmstudio  "$logical" "lmstudio-alt" "$lms_gguf_id" ;;
        esac
        # Settle only after a cell that actually ran. A skipped cell (engine
        # not installed, model not eligible) costs nothing, and the default
        # mlx-serve-only run skips four of five cells per model — sleeping
        # through those added ~8 min of nothing to a six-model run.
        [[ "$CELL_RAN" -eq 1 ]] || continue
        stop_all_engines
        sleep "$SETTLE"
    done
done

stop_all_engines
lms unload --all >/dev/null 2>&1 || true

# ── Artifacts ──
DEPTH_NOTE="$([[ $FULL -eq 1 ]] && echo "llmprobe --bench-only --full (median of 3/rung, to 64k)" \
                                 || echo "llmprobe --bench-only (one run/rung, to 16k)")"
# Engine versions ride into the CSV so both charts can name the BUILD each bar
# came from — "+23% vs oMLX" ages into a claim about whatever oMLX is today.
# llmprobe's reports carry none, and each engine reports its own differently.
engine_versions() {
    local out=""
    local mine; mine="$("$BINARY" --version 2>/dev/null | grep -oE 'mlx-serve [0-9.]+' | awk '{print $2}')"
    [ -n "$mine" ] && out="mlx-serve=$mine"
    if [ -x "$OMLX_BIN" ]; then
        local v; v="$(defaults read /Applications/oMLX.app/Contents/Info.plist CFBundleShortVersionString 2>/dev/null)"
        [ -n "$v" ] && out="$out omlx=$v"
    fi
    if [ -x "$MTPLX_BIN" ]; then
        local v; v="$("$MTPLX_BIN" --version 2>&1 | head -1 | awk '{print $2}')"
        [ -n "$v" ] && out="$out mtplx=$v"
    fi
    if [ "$INCLUDE_LMSTUDIO" -eq 1 ]; then
        local v; v="$(defaults read "/Applications/LM Studio.app/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null)"
        [ -n "$v" ] && out="$out lmstudio=$v"
    fi
    echo "$out"
}

if ! python3 "$SCRIPT_DIR/bench_csv.py" "$JSON_DIR" --out "$CSV_OUT" \
    --engines "$(engine_versions)" \
    --note "$(date '+%Y-%m-%d') · $DEPTH_NOTE · engines booted with shipping defaults"; then
    echo "no CSV written — nothing measured" >&2
    exit 1
fi

if [[ "$NO_PLOT" -eq 1 ]]; then
    echo "(charts skipped — --no-plot)"
    exit 0
fi

# The ladder is per model. When the run didn't include the default ladder model
# (a `--only` dev loop), plot whatever DID run rather than erroring out at the
# end of a long bench — but never silently blend several models into one lane.
LADDER_ARGS=()
if grep -q "^$LADDER_MODEL|" "$CSV_OUT"; then
    LADDER_ARGS=(--model "$LADDER_MODEL")
else
    only_model="$(grep -v '^#' "$CSV_OUT" | tail -n +2 | cut -d'|' -f1 | sort -u)"
    if [[ "$(echo "$only_model" | grep -c .)" -eq 1 ]]; then
        echo "ladder: $LADDER_MODEL not in this run, using $only_model" >&2
        LADDER_ARGS=(--model "$only_model")
    else
        echo "ladder: $LADDER_MODEL not in this run and several models are — pass --ladder-model" >&2
    fi
fi

mkdir -p "$(dirname "$PNG_BAR")" "$(dirname "$PNG_LADDER")"
python3 "$SCRIPT_DIR/plot_vs_lmstudio_omlx.py" "$CSV_OUT" "$PNG_BAR"
python3 "$SCRIPT_DIR/plot_mtp_ladder.py" "$CSV_OUT" "$PNG_LADDER" "${LADDER_ARGS[@]+"${LADDER_ARGS[@]}"}"

echo
echo "=== CSV    $CSV_OUT"
echo "=== charts $PNG_BAR"
echo "           $PNG_LADDER"
