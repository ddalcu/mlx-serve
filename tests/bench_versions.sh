#!/bin/bash
# bench_versions.sh — the SAME engine, two builds. Shipped app binary vs the
# working-tree build, same models, same flags, llmprobe measuring both.
#
# This is not a cross-engine bench: `tests/bench.sh` owns that, and its CSVs
# are the release record. This one answers a narrower question — "did the tree
# get faster than what shipped, and where" — so it takes the same measurement
# layer (llmprobe --bench-only) and varies ONLY the binary.
#
#   ./tests/bench_versions.sh                    # every model, both arms
#   ./tests/bench_versions.sh --only muse-30b    # one row
#   ./tests/bench_versions.sh --resume <tag>     # pick up where a run stopped
#   ./tests/bench_versions.sh --skip inkling-small-2bit   # leave a row for later
#   ./tests/bench_versions.sh --list             # matrix + what each row proves
#
# PAUSE / RESUME. Every (model, arm) is a UNIT recorded in state.tsv the moment
# it finishes. `touch <rundir>/PAUSE` and the runner stops at the next unit
# boundary and waits for the file to disappear — so a model swap, a meeting, or
# a reboot never costs more than the unit in flight. Ctrl-C is safe: rerun with
# `--resume <tag>` and completed units are skipped, not re-measured.
#
# WHY THE LADDER IS THE CHART. Both binaries are byte-identical at short
# context; the sliding-window block trim only pays once the cache is past the
# window. The headline decode/prefill/TTFT cells will show ~nothing and that is
# CORRECT — read the rung columns.
#
# ARM LABELLING. Both builds report the same --version string, so the arms are
# named for their BUILD (`shipped` = the .app, `dev` = zig-out), and the CSV
# header records the binary mtimes. Never label these by version number.
#
# THERMAL. The two arms of a model run back-to-back in one session, because a
# ratio is only honest within a session (a cold box and a soaked box are
# different machines). Which arm goes FIRST alternates by row, so the warm-up
# advantage lands on `shipped` half the time and `dev` the other half instead
# of systematically favouring one.
#
# Requirements: node (npx), python3 + matplotlib, curl, and a ReleaseFast build
# (`zig build -Doptimize=ReleaseFast` — Debug is 2-4x slower = a fake win).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEV_BIN="${DEV_BIN:-$ROOT/zig-out/bin/mlx-serve}"
SHIPPED_BIN="${SHIPPED_BIN:-/Applications/MLX Core.app/Contents/MacOS/mlx-serve}"
LLMPROBE="${LLMPROBE:-npx --yes llmprobe@latest}"
PORT="${BENCH_PORT:-11260}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/claude-tmp/bench-versions}"

ONLY=""
SKIP=""
RESUME_TAG=""
LIST=0
FULL=0
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;   # comma list of logical names
        --resume)  RESUME_TAG="$2"; shift 2 ;;
        --list)    LIST=1; shift ;;
        --full)    FULL=1; shift ;;   # ladder to 64k, median-of-3 per rung
        --quick)   QUICK=1; shift ;;  # one rung (8k), one run: the iteration smoke
        --port)    PORT="$2"; shift 2 ;;
        -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'; DIM='\033[2m'; NC='\033[0m'

# ── Model matrix ──
# logical|path|spec_args|note
#
# `spec_args` is EMPTY wherever shipping defaults already do the right thing:
# PLD is on by default, an in-dir `mtp/` or `drafter/` sidecar auto-loads, and
# a Gemma-4 drafter is deliberately default-OFF on MoE targets (it regresses
# there; PLD wins). The only two rows that need a flag are the ones whose
# sidecar lives OUTSIDE the model dir, or whose default is off by policy.
# Both arms get byte-identical flags, and the shipped binary was checked to
# accept every one of them — an unknown flag is REJECTED by mlx-serve's arg
# loop, so a drifted flag fails loudly instead of silently measuring a
# different configuration. (It already caught one: a model root with a SPACE in
# it — /Volumes/G Drive SSD — split a --drafter path into three words and the
# server refused to boot rather than quietly benching without the sidecar.
# Quote any path inside spec_args; it is re-split with `eval` below.)
MD="$HOME/.mlx-serve/models"
SSD="/Volumes/G Drive SSD/models"

TARGETS=(
  # ── sliding-window archs: where the trim pays ──
  "muse-30b-4bit|$MD/ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit||52 layers @ sw 2048; in-dir drafter/ = DFlash"
  "gemma4-26b-a4b-4bit|$SSD/mlx-community/gemma-4-26b-a4b-it-4bit||MoE, 25/30 sliding @ sw 1024; PLD"
  "gemma4-e4b-4bit|$SSD/mlx-community/gemma-4-e4b-it-4bit|--drafter '$SSD/mlx-community/gemma-4-E4B-it-assistant-bf16'|dense, 35/42 sliding @ sw 512; sidecar drafter"
  "laguna-xs-nvfp4|$SSD/poolside/Laguna-XS-2.1-NVFP4-mlx||serial coder, 30/40 sliding @ sw 512; prefill-chunk win"
  "inkling-small-2bit|$SSD/mlx-community/Inkling-Small-mlx-2bit||serial MoE, 35/42 sliding @ sw 512; RelativeLogits bias"
  # ── controls: no sliding layers, so these must come out FLAT ──
  "qwen38-27b-4bit|$SSD/ddalcu/Qwen3.8-27B-MLX-Serve-4bit||CONTROL: no sliding; in-checkpoint MTP head; the round-cost table's home cell"
  "lfm2-2.6b-nvfp4|$SSD/mlx-community/LFM2.5-2.6B-nvfp4||CONTROL: hybrid conv+full attn, no sliding; cheap smoke"
)

if [[ "$LIST" -eq 1 ]]; then
    printf "%-22s %-9s %s\n" "MODEL" "ON DISK" "WHAT IT PROVES"
    for t in "${TARGETS[@]}"; do
        IFS='|' read -r name path spec note <<<"$t"
        if [[ -d "$path" ]]; then sz=$(du -sh "$path" 2>/dev/null | cut -f1); else sz="MISSING"; fi
        printf "%-22s %-9s %s\n" "$name" "$sz" "$note"
        [[ -n "$spec" ]] && printf "%-22s %-9s %s\n" "" "" "  spec: $spec"
    done
    exit 0
fi

# ── Run directory + state ──
if [[ -n "$RESUME_TAG" ]]; then
    TAG="$RESUME_TAG"
    RUN_DIR="$RUNS_ROOT/$TAG"
    [[ -d "$RUN_DIR" ]] || { echo -e "${RED}no such run: $RUN_DIR${NC}" >&2; exit 1; }
    echo -e "${YEL}resuming${NC} $TAG"
else
    TAG="$(date +%Y%m%d-%H%M%S)"
    RUN_DIR="$RUNS_ROOT/$TAG"
fi
JSON_DIR="$RUN_DIR/json"; LOG_DIR="$RUN_DIR/logs"
STATE="$RUN_DIR/state.tsv"; PAUSE_FILE="$RUN_DIR/PAUSE"
mkdir -p "$JSON_DIR" "$LOG_DIR"; touch "$STATE"

[[ -x "$DEV_BIN" ]] || { echo -e "${RED}no dev binary at $DEV_BIN${NC} — zig build -Doptimize=ReleaseFast" >&2; exit 1; }
[[ -x "$SHIPPED_BIN" ]] || { echo -e "${RED}no shipped binary at $SHIPPED_BIN${NC}" >&2; exit 1; }

bin_for()   { [[ "$1" == "dev" ]] && echo "$DEV_BIN" || echo "$SHIPPED_BIN"; }
bin_stamp() { stat -f "%Sm" -t "%Y-%m-%dT%H:%M" "$1" 2>/dev/null; }

# ── Engine lifecycle ──
# The kill list and the wait list must name the SAME port, or every stop burns
# the full timeout (11 min/run when this was last broken).
stop_engine() {
    pkill -f "mlx-serve --serve" 2>/dev/null
    for _ in $(seq 1 40); do
        lsof -ti tcp:"$PORT" >/dev/null 2>&1 || return 0
        sleep 1
    done
    echo -e "  ${YEL}warn${NC}: port $PORT still busy after 40s" >&2
}
trap 'stop_engine' EXIT

wait_if_paused() {
    [[ -f "$PAUSE_FILE" ]] || return 0
    echo -e "${YEL}PAUSED${NC} — remove $PAUSE_FILE to continue (state is safe; Ctrl-C + --resume $TAG also works)"
    while [[ -f "$PAUSE_FILE" ]]; do sleep 5; done
    echo -e "${GRN}resuming${NC}"
}

unit_done() { grep -qF "$(printf '%s\t%s\tOK' "$1" "$2")" "$STATE"; }

# run_unit <logical> <path> <spec_args> <arm>
run_unit() {
    local name="$1" path="$2" spec="$3" arm="$4"
    local bin; bin="$(bin_for "$arm")"
    local out="$JSON_DIR/${name}__${arm}__default.json"
    local slog="$LOG_DIR/${name}__${arm}.server.log"
    local plog="$LOG_DIR/${name}__${arm}.probe.log"

    echo -e "  ${DIM}[$arm]${NC} booting $(basename "$bin") ($(bin_stamp "$bin"))"
    stop_engine
    # spec_args needs word-splitting to separate flag from value, but a bare
    # $spec also splits INSIDE a quoted path. eval re-splits honouring the
    # quotes in the matrix, so a path with a space survives as one word.
    # The [@]+ guard is not optional: macOS ships bash 3.2, where expanding an
    # EMPTY array as "${a[@]}" under `set -u` is an unbound-variable error. Most
    # rows have no spec args, so without it every default-config model fails to
    # boot. Same idiom as bench.sh's LADDER_ARGS.
    local -a spec_arr=()
    [[ -n "$spec" ]] && eval "spec_arr=($spec)"
    "$bin" --serve --model "$path" --port "$PORT" "${spec_arr[@]+"${spec_arr[@]}"}" >"$slog" 2>&1 &
    local pid=$!

    local up=0
    for _ in $(seq 1 900); do
        curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    if [[ "$up" != "1" ]]; then
        echo -e "  ${RED}FAIL${NC} $name/$arm: server never came up"
        tail -15 "$slog" | sed 's/^/      /'
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
        printf '%s\t%s\tFAIL\n' "$name" "$arm" >>"$STATE"
        return 1
    fi

    # The id mlx-serve registered, straight from the horse's mouth — never
    # guessed from the path, so a discovery rename can't silently probe the
    # wrong model (or, worse, llmprobe's "first model" fallback).
    local mid
    mid=$(curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" \
          | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(d[0]['id'] if d else '')" 2>/dev/null)
    if [[ -z "$mid" ]]; then
        echo -e "  ${RED}FAIL${NC} $name/$arm: /v1/models empty"
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
        printf '%s\t%s\tFAIL\n' "$name" "$arm" >>"$STATE"
        return 1
    fi

    local depth=(--bench-only)
    [[ "$FULL" -eq 1 ]] && depth+=(--full)
    [[ "$QUICK" -eq 1 ]] && depth+=(--rungs 8k --runs 1)
    echo -e "  ${DIM}[$arm]${NC} probing $mid"
    local rc=0
    # shellcheck disable=SC2086
    $LLMPROBE "localhost:$PORT" -m "$mid" "${depth[@]}" --save "$out" --no-save \
        >"$plog" 2>&1 || rc=$?

    # Speculation actually engaging is a property of the RUN, not of the flags
    # we passed — record what the server said it did so a flat row can be told
    # apart from a row where spec silently never armed.
    local eng
    eng=$(grep -ohE "\[spec-stats\] mode=[a-z]+" "$slog" | sort -u | tr '\n' ' ')
    [[ -n "$eng" ]] && echo -e "  ${DIM}[$arm]${NC} spec: $eng"
    grep -q "\[sliding\] block trim engaged" "$slog" \
        && echo -e "  ${DIM}[$arm]${NC} sliding block trim: ENGAGED"

    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    stop_engine

    if [[ $rc -ne 0 || ! -s "$out" ]]; then
        echo -e "  ${RED}FAIL${NC} $name/$arm: llmprobe rc=$rc"
        tail -10 "$plog" | sed 's/^/      /'
        printf '%s\t%s\tFAIL\n' "$name" "$arm" >>"$STATE"
        return 1
    fi
    printf '%s\t%s\tOK\n' "$name" "$arm" >>"$STATE"
    echo -e "  ${GRN}ok${NC} $name/$arm"
    return 0
}

# ── Drive ──
echo "=== mlx-serve build A/B ==="
echo "  shipped: $SHIPPED_BIN  ($(bin_stamp "$SHIPPED_BIN"))"
echo "  dev:     $DEV_BIN  ($(bin_stamp "$DEV_BIN"))"
echo "  run dir: $RUN_DIR"
echo "  depth:   $([[ $FULL -eq 1 ]] && echo '--full (median of 3/rung, to 64k)' || ([[ $QUICK -eq 1 ]] && echo '--quick (8k rung, one run)' || echo 'one run/rung, to 16k'))"
echo "  pause:   touch $RUN_DIR/PAUSE"
echo

idx=0
for t in "${TARGETS[@]}"; do
    IFS='|' read -r name path spec note <<<"$t"
    [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
    if [[ -n "$SKIP" && ",$SKIP," == *",$name,"* ]]; then
        # Deliberately left for a later session. Nothing is written to state,
        # so a plain --resume still picks it up untouched.
        echo -e "${YEL}skip${NC} $name: excluded by --skip"
        continue
    fi
    if [[ ! -d "$path" ]]; then
        echo -e "${YEL}skip${NC} $name: $path not found"
        continue
    fi
    idx=$((idx + 1))

    # Alternate which arm is measured first so the warm-up advantage does not
    # land on the same build every row.
    if (( idx % 2 == 1 )); then arms=(shipped dev); else arms=(dev shipped); fi

    echo "== $name =="
    echo -e "  ${DIM}$note${NC}"
    for arm in "${arms[@]}"; do
        if unit_done "$name" "$arm"; then
            echo -e "  ${DIM}[$arm] already done — skipping${NC}"
            continue
        fi
        wait_if_paused
        run_unit "$name" "$path" "$spec" "$arm" || true
    done
    echo
done

# ── Fold + plot ──
CSV="$RUN_DIR/versions-$TAG.csv"
NOTE="$(date +%Y-%m-%d) · llmprobe --bench-only $([[ $FULL -eq 1 ]] && echo '--full (median of 3/rung, to 64k)' || echo '(one run/rung, to 16k)') · same engine, two builds · shipping defaults"
python3 "$SCRIPT_DIR/bench_csv.py" "$JSON_DIR" --out "$CSV" \
    --note "$NOTE" \
    --engines "shipped=$(bin_stamp "$SHIPPED_BIN") dev=$(bin_stamp "$DEV_BIN")" \
    || { echo -e "${RED}CSV fold failed${NC}" >&2; exit 1; }

PNG="$RUN_DIR/mlx-serve-build-ab-$TAG.png"
python3 "$SCRIPT_DIR/plot_version_ab.py" "$CSV" "$PNG" || echo -e "${YEL}chart failed (CSV is still good)${NC}" >&2

echo
echo -e "${GRN}done${NC}"
echo "  csv:   $CSV"
echo "  chart: $PNG"
echo "  state: $STATE"
# A unit that FAILED and was later retried OK is not a failure — state.tsv is
# append-only, so count units with NO ok record rather than counting FAIL lines.
awk -F'\t' -v tag="$TAG" '
    $3=="OK"   { ok[$1 SUBSEP $2]=1 }
    $3=="FAIL" { bad[$1 SUBSEP $2]=1 }
    END { for (k in bad) if (!(k in ok)) n++
          if (n) printf "  %d unit(s) FAILED — rerun: ./tests/bench_versions.sh --resume %s\n", n, tag }' "$STATE"
