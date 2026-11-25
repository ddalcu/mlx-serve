#!/bin/bash
# test_first_request_compiles.sh — every load of a model warms its Metal kernels with requests of its own, so the
# prompts someone sends add next to nothing to Metal's shader cache. Nothing kept from an earlier load decides it:
# a load after another under other switches, flags or pack contents serves its own warm requests and compiles its own
# kernels before (--warmup-full) or after (default, when idle) it reports ready. The server runs as a copy of the
# binary under its own bundle id, so its shader cache starts EMPTY, and the cache index (`libraries.list`) is sized
# around the test's requests. A copy with --no-warmup-eager is the control: there the first prompts compile what the
# load did not, and the test must see it.
#
#   ./tests/test_first_request_compiles.sh [port]   # macOS only. BINARY, WARM_TEST_MODEL=<org>/<repo> or <pack dir>
#   WARM_TEST_DSV41=<pack>        also the DeepSeek-V4.1 cases: load X, then load Y with --warmup-full; Y's requests
#                                 compile no more than they do on Y's own first load, for X -> Y = DSpark off -> on,
#                                 NVFP4 -> the MXFP4 view, PF_GU_SEG=0 -> on, --no-mtp -> MTP, and the pack replaced
#                                 under the same path (WARM_TEST_DSV41_ALT=<another pack>); and a prompt at the load's
#                                 widest pass compiles next to nothing after --warmup-full
#   WARM_TEST_DSV41_FLAGS         the DeepSeek server's flags (default --ctx-size 32768)
#   WARM_TEST_OLD_BINARY=<bin>    also runs the DeepSeek cases on another build and reports which ones it fails
#   WARM_TEST_HOLD=<dir>          move the test's scratch, copies and shader caches there instead of deleting them
#
# Hermetic twins: `promptIds`, the plans' forks and ladder, `widestRung`, `the full plan`, `nextStep`, the idle
# warm-up's tests (warm_requests.zig), `ramTwin`, `submitNow`.

set -u
source "$(dirname "$0")/lib/server.sh"
source "$(dirname "$0")/_lib_models.sh"

PORT="${1:-11391}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
HOLD="${WARM_TEST_HOLD:-}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

[ "$(uname)" = Darwin ] || { echo -e "${YELLOW}SKIP${NC} first-request compiles: Metal only."; exit 0; }
[ -x "$BINARY" ] || { echo -e "${RED}FAIL${NC} no $BINARY: build ReleaseFast first"; exit 1; }
# WARM_TEST_MODEL: an <org>/<repo> to look up, or a pack's directory.
if [ -n "${WARM_TEST_MODEL:-}" ] && [ -d "$WARM_TEST_MODEL" ]; then MODEL="$WARM_TEST_MODEL"
else MODEL=$(find_fitting_model ${WARM_TEST_MODEL:-mlx-community/gemma-4-e4b-it-4bit ddalcu/Qwen3.8-27B-MLX-Serve-4bit}) || MODEL=""; fi
# Part 1's server flags (a big pack needs its context set: WARM_TEST_FLAGS="--ctx-size 32768").
P1_FLAGS=${WARM_TEST_FLAGS:-}
DSV41="${WARM_TEST_DSV41:-}"
[ -n "$MODEL" ] || [ -n "$DSV41" ] || { echo -e "${YELLOW}SKIP${NC} first-request compiles: no test model on this box."; exit 0; }

# Metal keys its on-disk cache on the bundle id every build embeds: a copy under another id of the same length
# starts with none.
ID_OLD=com.dalcu.mlx-core.mlx-serve
ID_NEW=com.dalcu.mlx-cold.mlx-serve
CACHE="$(getconf DARWIN_USER_CACHE_DIR)$ID_NEW"
SCRATCH="$HOME/claude-tmp/test-first-request-$$"
SRV=""
mkdir -p "$SCRATCH"
[ -n "$HOLD" ] && mkdir -p "$HOLD"
# away <path>: the test's own file gone, moved into WARM_TEST_HOLD when it is set.
N_AWAY=0
away() {
    [ -e "$1" ] || [ -L "$1" ] || return 0
    if [ -n "$HOLD" ]; then N_AWAY=$((N_AWAY + 1)); mv "$1" "$HOLD/$(basename "$1").$$.$N_AWAY"; else rm -rf "$1"; fi
}
COPIES=()
cleanup() {
    srv_stop "$SRV"
    for c in "${COPIES[@]}"; do away "$c"; done
    away "$CACHE"; away "$SCRATCH"
}
trap cleanup EXIT

# copy_of <binary> <name>: the binary under the test's bundle id, signed, in its own path.
copy_of() {
    local dst; dst="$(dirname "$BINARY")/mlx-serve-coldcache.$$.$2"
    python3 - "$1" "$dst" "$ID_OLD" "$ID_NEW" <<'PY' || { echo -e "${RED}FAIL${NC} the binary does not embed its bundle id once" >&2; exit 1; }
import os, sys
src, dst, old, new = sys.argv[1], sys.argv[2], sys.argv[3].encode(), sys.argv[4].encode()
data = open(src, "rb").read()
assert data.count(old) == 1 and len(old) == len(new)
open(dst, "wb").write(data.replace(old, new))
os.chmod(dst, 0o755)
PY
    codesign -f -s - "$dst" 2>/dev/null
    COPIES+=("$dst")
    echo "$dst"
}

READY="Model ready (loaded on inference thread)"
DONE="Warm requests complete"
line_of() { grep -a -n -m1 -F "$1" "$2" | cut -d: -f1; }

# compiled <binary> <model> <log> <keep|empty> <idle|ready> <short|wide> <env words> [flag ...]: bytes the shader
# library index gained over the test's requests, the shader cache emptied first or kept as the last boot left it.
# `idle` waits for the idle warm-up to end before the first prompt; `ready` sends it as soon as /health answers.
# `short`: a short chat, a short prompt and another question under it, then a long prompt and another question under
# that (two numbers: short + long). `wide`: one prompt at WIDE_TOKENS (one number).
compiled() {
    local bin="$1" model="$2" log="$3" cache="$4" start="$5" kind="$6" envw="$7"; shift 7
    [ "$cache" = empty ] && away "$CACHE"
    away "$SCRATCH/home"; mkdir -p "$SCRATCH/home"
    # A big pack's last server can still be giving its memory back when the next one loads: a load refused for memory
    # (`InsufficientMemory`) is started again after a pause, up to 20 times; any other failure ends the test.
    local try
    for try in $(seq 1 20); do
        env HOME="$SCRATCH/home" $envw "$bin" --serve --model "$model" --port "$PORT" --host 127.0.0.1 "$@" >"$log" 2>&1 &
        SRV=$!
        srv_wait_health "$BASE" 2400 0.5 "$SRV" && break
        srv_stop "$SRV"; SRV=""
        grep -a -q "InsufficientMemory" "$log" || { echo -e "${RED}FAIL${NC} the server never came up" >&2; tail -5 "$log" >&2; exit 1; }
        [ "$try" = 20 ] && { echo -e "${RED}FAIL${NC} the server never came up: refused for memory 20 times" >&2; tail -5 "$log" >&2; exit 1; }
        sleep 15
    done
    if [ "$start" = idle ]; then
        for _ in $(seq 1 1200); do grep -a -q -F "$DONE" "$log" && break; sleep 0.5; done
        grep -a -q -F "$DONE" "$log" || { echo -e "${RED}FAIL${NC} the idle warm-up never ended" >&2; tail -5 "$log" >&2; srv_stop "$SRV"; exit 1; }
    fi
    python3 - "$BASE" "$CACHE" "$kind" "${WIDE_TOKENS:-0}" <<'PY' || { echo -e "${RED}FAIL${NC} a request failed, or the first one found a prompt cached" >&2; srv_stop "$SRV"; exit 1; }
import glob, json, os, sys, urllib.request
def index():
    return sum(os.path.getsize(p) for p in glob.glob(sys.argv[2] + "/**/libraries.list", recursive=True))
def ask(text, n):
    body = dict(model="x", messages=[{"role": "user", "content": text}], max_tokens=n, temperature=0)
    req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))["usage"]
def doc(lines):
    return " ".join(f"[{i}] The archive records the following entry." for i in range(lines))
if sys.argv[3] == "wide":
    # The tokens a line, from two short prompts before the reading starts; then one prompt just under the widest pass.
    a = ask(doc(50) + "\n\nReply with just the word: OK.", 1)["prompt_tokens"]
    b = ask(doc(150) + "\n\nReply with just the word: OK.", 1)["prompt_tokens"]
    per = max(1.0, (b - a) / 100)
    lines = max(1, int((int(sys.argv[4]) - (a - 50 * per)) / per))
    start = index()
    u = ask(doc(lines) + "\n\nReply with just the word: OK.", 1)
    print(index() - start, u.get("prompt_tokens", 0))
    sys.exit(0)
start = index()
first = ask("Write a Python function that reverses a linked list, with comments.", 64)
# The warm requests left nothing behind in the model's prefix cache.
assert (first.get("prompt_tokens_details") or {}).get("cached_tokens", 0) == 0, first
ask(doc(20) + "\n\nReply with just the word: OK.", 8)
ask(doc(20) + "\n\nIn two sentences, say what the text above consists of.", 64)
short = index()
ask(doc(180) + "\n\nReply with just the word: OK.", 8)
ask(doc(180) + "\n\nIn two sentences, say what the text above consists of.", 64)
print(short - start, index() - short)
PY
    srv_stop "$SRV"; SRV=""
}

FAIL=0
fail() { echo -e "${RED}FAIL${NC} $1"; FAIL=1; }

# ── part 1: every load warms, by default when idle and with --warmup-full before ready ──
if [ -n "$MODEL" ]; then
    echo "  part 1 on $MODEL"
    COPY=$(copy_of "$BINARY" idle); FULL=$(copy_of "$BINARY" full); CONTROL=$(copy_of "$BINARY" control)
    read -r IDLE_S IDLE_L <<< "$(compiled "$COPY" "$MODEL" "$SCRATCH/idle.log" empty idle short "" $P1_FLAGS)"
    read -r AGAIN_S AGAIN_L <<< "$(compiled "$COPY" "$MODEL" "$SCRATCH/idle2.log" keep idle short "" $P1_FLAGS)"
    read -r FULL_S FULL_L <<< "$(compiled "$FULL" "$MODEL" "$SCRATCH/full.log" empty ready short "" --warmup-full $P1_FLAGS)"
    read -r FULL2_S FULL2_L <<< "$(compiled "$FULL" "$MODEL" "$SCRATCH/full2.log" keep ready short "" --warmup-full $P1_FLAGS)"
    read -r COLD_S COLD_L <<< "$(compiled "$CONTROL" "$MODEL" "$SCRATCH/cold.log" empty ready short "" --no-warmup-eager $P1_FLAGS)"
    [ -n "$IDLE_L" ] && [ -n "$AGAIN_L" ] && [ -n "$FULL_L" ] && [ -n "$FULL2_L" ] && [ -n "$COLD_L" ] || { echo -e "${RED}FAIL${NC} an arm gave no reading"; exit 1; }
    echo "  shader-library index bytes gained, short prompts + long prompts:"
    echo "    default, after its idle warm-up $IDLE_S + $IDLE_L; its second load $AGAIN_S + $AGAIN_L"
    echo "    --warmup-full $FULL_S + $FULL_L; its second load $FULL2_S + $FULL2_L; --no-warmup-eager $COLD_S + $COLD_L"
    # Default: ready first, the warm requests after it. --warmup-full: the warm requests first. Every load serves them.
    for log in idle idle2; do
        [ "$(line_of "$READY" "$SCRATCH/$log.log")" -lt "$(line_of "$DONE" "$SCRATCH/$log.log")" ] || fail "$log: the default load did not serve its warm requests after it reported ready"
    done
    for log in full full2; do
        d=$(line_of "$DONE" "$SCRATCH/$log.log")
        [ -n "$d" ] && [ "$d" -lt "$(line_of "$READY" "$SCRATCH/$log.log")" ] || fail "$log: --warmup-full reported ready before it had served its warm requests"
    done
    grep -a -q -F "$DONE" "$SCRATCH/cold.log" && fail "cold: --no-warmup-eager served warm requests"
    # The control proves the reading: without the warm requests these prompts compile shaders, and plenty of them.
    COLD=$(( COLD_S + COLD_L ))
    [ "$COLD_S" -ge 500 ] && [ "$COLD" -ge 1000 ] || fail "the control compiled nearly nothing ($COLD_S + $COLD_L bytes): the reading is blind"
    [ $(( IDLE_S * 3 )) -le "$COLD_S" ] || fail "after the idle warm-up the short prompts compiled $IDLE_S bytes of index, more than a third of the unwarmed $COLD_S"
    [ $(( AGAIN_S * 3 )) -le "$COLD_S" ] || fail "the default's second load compiled $AGAIN_S bytes of index on the short prompts"
    [ $(( (FULL_S + FULL_L) * 3 )) -le "$COLD" ] || fail "after --warmup-full the prompts compiled $(( FULL_S + FULL_L )) bytes of index, more than a third of the unwarmed $COLD"
    [ $(( (FULL2_S + FULL2_L) * 3 )) -le "$COLD" ] || fail "the second load after --warmup-full compiled $(( FULL2_S + FULL2_L )) bytes of index"
fi

# ── part 2: DeepSeek-V4.1, a load after another under other kernel arms; the widest pass ──
# case_xy <bin> <name> <X env> <X pack> <Y env> <Y pack> [Y flag ...]: R = what Y's requests compile on Y's own
# first load (an empty cache, then Y's full warm-up); G = what they compile when Y loads after X (X's warm-up and
# requests first). Y's warm-up compiles Y's kernels whatever X left, so G <= R. A load that skipped its warm-up because
# of X compiles Y's plan kernels inside its requests: G > R.
DS_FLAGS=${WARM_TEST_DSV41_FLAGS:---ctx-size 32768}
DS_RESULTS="$SCRATCH/ds-results.txt"
: > "$DS_RESULTS"
# PACK_X / PACK_Y: the pack `repoint` puts under one path before X's load and before Y's (Y's reference included).
case_xy() {
    local bin="$1" name="$2" xenv="$3" xpack="$4" yenv="$5" ypack="$6"; shift 6
    local yflags="$*" s l r g
    [ -n "${PACK_Y:-}" ] && repoint "$PACK_Y"
    read -r s l <<< "$(compiled "$bin" "$ypack" "$SCRATCH/$name-ref.log" empty ready short "$yenv" $DS_FLAGS --warmup-full $yflags)"
    r=$(( s + l ))
    [ -n "${PACK_X:-}" ] && repoint "$PACK_X"
    compiled "$bin" "$xpack" "$SCRATCH/$name-x.log" empty ready short "$xenv" $DS_FLAGS --warmup-full ${XFLAGS:-} >/dev/null
    [ -n "${PACK_Y:-}" ] && repoint "$PACK_Y"
    read -r s l <<< "$(compiled "$bin" "$ypack" "$SCRATCH/$name-y.log" keep ready short "$yenv" $DS_FLAGS --warmup-full $yflags)"
    g=$(( s + l ))
    local served=no; grep -a -q -F "$DONE" "$SCRATCH/$name-y.log" && served=yes
    local verdict=PASS; { [ "$served" = yes ] && [ "$g" -le "$r" ]; } || verdict=FAIL
    echo "$name: Y served its warm requests $served; Y's requests compiled $g bytes after X, $r on Y's own first load: $verdict" >> "$DS_RESULTS"
    echo "$verdict"
}

# The pack under one path, its files links to <pack>: `repoint <pack>` moves the old links aside and makes new ones.
SAMEPATH="$SCRATCH/pack-under-one-path"
repoint() {
    away "$SAMEPATH"; mkdir -p "$SAMEPATH"
    local f; for f in "$1"/*; do ln -s "$f" "$SAMEPATH/$(basename "$f")"; done
}

ds_cases() {
    local bin="$1" tag="$2" v
    v=$(case_xy "$bin" "$tag-dspark" "MLX_SERVE_DSV4_DSPARK=0" "$DSV41" "" "$DSV41"); [ "$v" = PASS ] || DS_FAILED+=("$tag dspark")
    v=$(case_xy "$bin" "$tag-mxview" "" "$DSV41" "MLX_SERVE_DSV41_MXFP4_VIEW=1" "$DSV41"); [ "$v" = PASS ] || DS_FAILED+=("$tag mxfp4-view")
    v=$(case_xy "$bin" "$tag-guseg" "MLX_SERVE_DSV41_PF_GU_SEG=0" "$DSV41" "" "$DSV41"); [ "$v" = PASS ] || DS_FAILED+=("$tag pf_gu_seg")
    v=$(XFLAGS=--no-mtp case_xy "$bin" "$tag-nomtp" "" "$DSV41" "" "$DSV41"); [ "$v" = PASS ] || DS_FAILED+=("$tag no-mtp")
    if [ -n "${WARM_TEST_DSV41_ALT:-}" ]; then
        v=$(PACK_X="$DSV41" PACK_Y="$WARM_TEST_DSV41_ALT" case_xy "$bin" "$tag-pack" "" "$SAMEPATH" "" "$SAMEPATH"); [ "$v" = PASS ] || DS_FAILED+=("$tag replaced-pack")
    fi
}

if [ -n "$DSV41" ]; then
    echo "  part 2 on $DSV41 ($DS_FLAGS)"
    DS_FAILED=()
    NEW=$(copy_of "$BINARY" ds41)
    ds_cases "$NEW" new
    NEW_FAILED=("${DS_FAILED[@]+"${DS_FAILED[@]}"}")
    # The widest pass: after --warmup-full, one prompt just under the load's widest pass compiles next to nothing
    # against the same prompt on an unwarmed load.
    # The width the cases' --warmup-full loads named (the same flags and pack).
    PASS_ROWS=$(grep -a -o "widest pass [0-9]* rows" "$SCRATCH/new-dspark-ref.log" | head -1 | grep -o "[0-9]*")
    [ -n "$PASS_ROWS" ] && [ "$PASS_ROWS" -gt 0 ] || fail "the --warmup-full load named no widest pass"
    WIDE_TOKENS=$(( ${PASS_ROWS:-0} - 600 ))
    read -r W_WARM W_TOK <<< "$(WIDE_TOKENS=$WIDE_TOKENS compiled "$NEW" "$DSV41" "$SCRATCH/wide.log" empty ready wide "" $DS_FLAGS --warmup-full)"
    read -r W_COLD _ <<< "$(WIDE_TOKENS=$WIDE_TOKENS compiled "$NEW" "$DSV41" "$SCRATCH/wide-cold.log" empty ready wide "" $DS_FLAGS --no-warmup-eager)"
    echo "    widest pass $PASS_ROWS rows: a $W_TOK-token prompt compiled $W_WARM bytes after --warmup-full, $W_COLD unwarmed"
    [ "${W_COLD:-0}" -ge 500 ] || fail "the unwarmed widest prompt compiled nearly nothing ($W_COLD bytes): the reading is blind"
    [ $(( ${W_WARM:-0} * 3 )) -le "${W_COLD:-0}" ] || fail "after --warmup-full the widest-pass prompt compiled $W_WARM bytes, more than a third of the unwarmed $W_COLD"
    for f in "${NEW_FAILED[@]+"${NEW_FAILED[@]}"}"; do fail "a DeepSeek-V4.1 load after another compiled its own kernels in its requests: $f"; done
    if [ -n "${WARM_TEST_OLD_BINARY:-}" ]; then
        DS_FAILED=()
        OLD=$(copy_of "$WARM_TEST_OLD_BINARY" old)
        ds_cases "$OLD" old
        read -r OW _ <<< "$(WIDE_TOKENS=$WIDE_TOKENS compiled "$OLD" "$DSV41" "$SCRATCH/old-wide.log" empty ready wide "" $DS_FLAGS --warmup-full)"
        echo "    the other build ($WARM_TEST_OLD_BINARY): fails ${DS_FAILED[*]:-none}; its widest-pass prompt compiled $OW bytes after --warmup-full"
    fi
    sed 's/^/    /' "$DS_RESULTS"
fi
[ "$FAIL" -eq 0 ] && echo -e "${GREEN}PASS${NC} every load warms its own kernels (when idle, or before ready with --warmup-full), whatever an earlier load compiled"
exit "$FAIL"
