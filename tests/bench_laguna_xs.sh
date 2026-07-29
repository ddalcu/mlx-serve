#!/bin/bash
# bench_laguna_xs.sh — Laguna XS 2.1 NVFP4 decode/prefill bench.
#
# Deliberately mirrors the mlx.fast challenge's frozen timed window so our
# numbers are directly comparable to the Layr-Labs/mlxfast-challenge tree:
# 512-token prompt prefill, 128 decode steps, temperature 0, and DECODE
# REPORTED IN ms/token (their unit) alongside tok/s.
#
# Two cells per run, never merged:
#   serial         PLD/drafter/MTP all off — their non-speculative rule. This
#                  is the apples-to-apples number and the only one that may be
#                  compared against their tree.
#   unconstrained  whatever mlx-serve's best configuration is (spec decode
#                  included). Reported separately; the challenge bans it.
#
# Timing comes from the SERVER's own `timings` object (prompt_ms / predicted_n
# / predicted_ms), measured around the forward passes — never from timing the
# HTTP response client-side, which folds in queueing and transport (the
# console's 937-tok/s lesson).
#
# Every request prepends a unique nonce so the prefix cache cannot serve the
# prefill; the script ASSERTS cached_n == 0 and fails the cell otherwise.
#
# Usage:
#   ./tests/bench_laguna_xs.sh                      # both cells, 3 runs, write CSV
#   ./tests/bench_laguna_xs.sh --runs 5
#   ./tests/bench_laguna_xs.sh --cells serial
#   ./tests/bench_laguna_xs.sh --label nvfp4-gatherqmv --note "3a on"
#   ./tests/bench_laguna_xs.sh --no-csv             # stdout only (A/B iteration)
#
# Env passthrough: any MLX_SERVE_* already exported is inherited by the server,
# so kill-switch A/Bs are `MLX_SERVE_FOO=0 ./tests/bench_laguna_xs.sh --no-csv`.
# ALWAYS rebuild with -Doptimize=ReleaseFast first: `zig build test` does NOT
# refresh zig-out/bin/mlx-serve.

set -uo pipefail

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${MODEL:-$HOME/.mlx-serve/models/poolside/Laguna-XS-2.1-NVFP4-mlx}"
PORT="${PORT:-8712}"
CTX="${CTX:-4096}"
MAX_TOKENS=128          # their decode window
TARGET_PROMPT_TOKENS=512 # their prefill window
RUNS=3
CELLS="serial unconstrained"
OUT_DIR="docs/perf-csvs"
WRITE_CSV=1
LABEL=""
NOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --cells) CELLS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --note) NOTE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --no-csv) WRITE_CSV=0; shift ;;
        -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }
[[ -x "$BINARY" ]] || { echo "binary not found: $BINARY (zig build -Doptimize=ReleaseFast)" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

HARDWARE="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

# ── Prompt: ~512 tokens of realistic code-review context. Built once so every
# run and every cell sees the SAME prefill work; a per-request nonce is
# prepended at send time to defeat the prefix cache.
build_prompt() {
    python3 - <<'PY'
body = []
body.append("Review the following Python module and list every correctness bug you find.\n\n")
body.append("```python\n")
# 4 blocks lands the rendered prompt at ~512 tokens on Laguna's tokenizer
# (calibrated live; the script prints prompt_n every run so drift is visible).
for i in range(4):
    body.append(f"""
def transform_batch_{i}(records, threshold={i}.5, normalize=True):
    \"\"\"Filter records above the threshold and normalize their weights.\"\"\"
    out = []
    total = 0.0
    for rec in records:
        if rec.get("score") > threshold:
            total += rec["score"]
            out.append(dict(rec))
    if normalize and total > 0:
        for rec in out:
            rec["weight"] = rec["score"] / total
    return out
""")
body.append("```\n")
print("".join(body))
PY
}
PROMPT_BODY="$(build_prompt)"

start_server() {
    # Serial vs unconstrained is selected PER REQUEST (the body always sends
    # explicit enable_pld/enable_drafter/enable_mtp), so one boot serves both
    # cells and neither pays a cold-load difference the other doesn't.
    "$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size "$CTX" \
        --log-level info >/tmp/bench_laguna_xs_server.log 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 480); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 0.5
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:" >&2; tail -20 /tmp/bench_laguna_xs_server.log >&2; return 1; }
    done
    echo "server never became healthy" >&2
    return 1
}

# one_request <cell> <nonce> -> "prefill_tps decode_tps ms_per_tok prompt_n cached_n predicted_n"
one_request() {
    local cell="$1" nonce="$2"
    local spec_on="false"
    [[ "$cell" == "unconstrained" ]] && spec_on="true"
    local body resp
    body=$(jq -nc \
        --arg content "run-$nonce. $PROMPT_BODY" \
        --argjson maxt "$MAX_TOKENS" \
        --argjson spec "$spec_on" \
        '{model:"laguna", messages:[{role:"user",content:$content}],
          max_tokens:$maxt, temperature:0, top_p:1, stream:false,
          enable_pld:$spec, enable_drafter:$spec, enable_mtp:$spec}')
    resp=$(curl -sf -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body") || { echo "REQFAIL"; return 1; }
    echo "$resp" | jq -r '
        .timings as $t |
        if $t == null then "NOTIMINGS" else
        [ (if $t.prompt_ms > 0 then ($t.prompt_n - $t.cached_n) * 1000 / $t.prompt_ms else 0 end),
          (if $t.predicted_ms > 0 then $t.predicted_n * 1000 / $t.predicted_ms else 0 end),
          (if $t.predicted_n > 0 then $t.predicted_ms / $t.predicted_n else 0 end),
          $t.prompt_n, $t.cached_n, $t.predicted_n ] | @tsv end'
}

median() { python3 -c "
import sys,statistics
vals=[float(x) for x in sys.argv[1:] if x]
print(f'{statistics.median(vals):.3f}' if vals else '0')" "$@"; }

echo "── Laguna XS 2.1 NVFP4 bench ──"
echo "   binary   : $BINARY"
echo "   model    : $MODEL"
echo "   hardware : $HARDWARE"
echo "   window   : ${TARGET_PROMPT_TOKENS}-token prompt / ${MAX_TOKENS} decode steps, temp 0"
echo "   runs     : $RUNS   cells: $CELLS"
env | grep -E '^MLX_SERVE_' | sed 's/^/   env      : /'
echo

start_server || exit 1

# Warmup (untimed): pays the first-request kernel JIT + any lazy weight eval.
one_request serial "warmup-$$" >/dev/null 2>&1

# bash 3.2 (macOS system bash) has no associative arrays — medians land in a
# temp file keyed by cell.
SUMMARY=$(mktemp)
FAILED=0
for cell in $CELLS; do
    pf=""; dc=""; mt=""; meta=""
    for r in $(seq 1 "$RUNS"); do
        line=$(one_request "$cell" "$cell-$$-$r")
        if [[ "$line" == "REQFAIL" || "$line" == "NOTIMINGS" || -z "$line" ]]; then
            echo "  !! $cell run $r failed ($line)" >&2; FAILED=1; continue
        fi
        read -r p d m pn cn dn <<<"$line"
        # The per-request nonce leads the user content, so everything after the
        # chat template's fixed header is uncacheable; a handful of cached
        # tokens IS that header. Anything more means the prefix cache served
        # real prompt work and the prefill number is fiction.
        if [[ "$cn" -gt $((pn / 8)) ]]; then
            echo "  !! $cell run $r: cached_n=$cn of $pn (prefix cache served the prompt) — number discarded" >&2
            FAILED=1; continue
        fi
        pf="$pf $p"; dc="$dc $d"; mt="$mt $m"; meta="$pn/$cn/$dn"
        printf "  %-14s run %d: prefill %8.1f tok/s | decode %6.2f tok/s = %6.3f ms/tok  (prompt %s, cached %s, gen %s)\n" \
            "$cell" "$r" "$p" "$d" "$m" "$pn" "$cn" "$dn"
    done
    [[ -z "$dc" ]] && continue
    echo "$cell $(median $pf) $(median $dc) $(median $mt) $meta" >> "$SUMMARY"
done

echo
echo "── medians (n=$RUNS) ──"
while read -r cell p d m meta; do
    printf "  %-14s prefill %8s tok/s | decode %6s tok/s = %6s ms/tok\n" "$cell" "$p" "$d" "$m"
done < "$SUMMARY"

if [[ "$WRITE_CSV" == "1" ]]; then
    mkdir -p "$OUT_DIR"
    csv="$OUT_DIR/laguna-xs-$(date +%Y-%m-%d).csv"
    if [[ ! -f "$csv" ]]; then
        echo "label|cell|model|prefill_tps|decode_tps|decode_ms_per_tok|prompt_toks|cached_toks|completion_toks|runs|hardware|notes" > "$csv"
    fi
    while read -r cell p d m meta; do
        IFS=/ read -r pn cn dn <<<"$meta"
        echo "${LABEL:-mlx-serve}|$cell|Laguna-XS-2.1-NVFP4-mlx|$p|$d|$m|$pn|$cn|$dn|$RUNS|$HARDWARE|$NOTE" >> "$csv"
    done < "$SUMMARY"
    echo
    echo "csv: $csv"
fi

rm -f "$SUMMARY"
exit "$FAILED"
