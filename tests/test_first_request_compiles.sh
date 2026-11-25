#!/bin/bash
# test_first_request_compiles.sh — every load of a model warms its Metal kernels with requests of its own, so the
# first prompts someone sends compile nothing. Nothing kept from an earlier load decides it: a load after another
# under other switches, flags or pack contents serves its own warm requests and compiles its own kernels before
# (--warmup-full) or after (default, when idle) it reports ready. The server runs as a copy of the binary under its
# own bundle id, so its shader cache starts EMPTY. The reading is time: each prompt shape is sent twice, two texts of
# the same size and no shared prefix, and a kernel compiled inside the first makes it slower than the second. A copy
# with --no-warmup-eager is the control: there the first prompts compile what the load did not, and the test must see
# it.
#
#   ./tests/test_first_request_compiles.sh [port]   # macOS only. BINARY, WARM_TEST_MODEL=<org>/<repo> or <pack dir>
#   WARM_TEST_DSV41=<pack>        also the DeepSeek-V4.1 cases: load X, then load Y with --warmup-full; Y serves its warm
#                                 requests and its first requests are no slower than their twins, for X -> Y = DSpark
#                                 off -> on, NVFP4 -> the MXFP4 view, PF_GU_SEG=0 -> on, --no-mtp -> MTP, and the pack
#                                 replaced under the same path (WARM_TEST_DSV41_ALT=<another pack>); and a prompt at the
#                                 load's widest pass is no slower than its twin after --warmup-full
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
# away <path>: the test's own file gone, moved into WARM_TEST_HOLD when it is set. Each move gets a new directory
# (mktemp), not a counter: away runs inside command substitutions, whose variable changes the caller never sees.
away() {
    [ -e "$1" ] || [ -L "$1" ] || return 0
    if [ -n "$HOLD" ]; then mv "$1" "$(mktemp -d "$HOLD/$(basename "$1").XXXXXX")/"; else rm -rf "$1"; fi
}
# The binary copies, one a line: a file, since copy_of also runs inside command substitutions.
COPIES="$SCRATCH/copies.list"
cleanup() {
    srv_stop "$SRV"
    if [ -f "$COPIES" ]; then while read -r c; do away "$c"; done < "$COPIES"; fi
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
    echo "$dst" >> "$COPIES"
    echo "$dst"
}

READY="Model ready (loaded on inference thread)"
DONE="Warm requests complete"
line_of() { grep -a -n -m1 -F "$1" "$2" | cut -d: -f1; }

# timed <binary> <model> <log> <keep|empty> <idle|ready> <short|wide> <env words> [flag ...]: the wall seconds of each
# request of a twin pair, first then second, the shader cache emptied first or kept as the last boot left it. `idle`
# waits for the idle warm-up to end before the first prompt; `ready` sends it as soon as /health answers. `short`:
# a short chat pair, then a long prompt pair (four numbers: short 1, short 2, long 1, long 2); `wide`: a pair just under
# WIDE_TOKENS (three numbers: 1, 2, the prompt's tokens). Each request generates 16 tokens, so decode's kernels run too.
timed() {
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
    python3 - "$BASE" "$kind" "${WIDE_TOKENS:-0}" <<'PY' || { echo -e "${RED}FAIL${NC} a request failed, or one found a prompt cached" >&2; srv_stop "$SRV"; exit 1; }
import json, sys, time, urllib.request
def ask(text, n):
    body = dict(model="x", messages=[{"role": "user", "content": text}], max_tokens=n, temperature=0)
    req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    u = json.load(urllib.request.urlopen(req, timeout=1800))["usage"]
    # Twins share no prefix past the chat template's few tokens: a cache hit would hide the forward the reading times.
    assert (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) < 32, u
    return time.monotonic() - t0, u
# Twins: one filler, opened by different words of the same size, so they are the same size and share no prefix.
# Zero-padded numbers keep every line the same tokens, so two short probes (their own openers) size a long text.
FILL = "The archive records the following entry."
OPEN = ("Archive copy.", "Ledger copy.", "Sizing probe one.", "Sizing probe two.")
def doc(lines, k):
    return OPEN[k] + " " + " ".join(f"[{i:05d}] {FILL}" for i in range(lines)) + "\n\nReply with just the word: OK."
if sys.argv[2] == "wide":
    # The tokens a line from two short probes (before the pair), then the pair at 98% of the target.
    _, a = ask(doc(100, 2), 1)
    _, b = ask(doc(300, 3), 1)
    per = max(1.0, (b["prompt_tokens"] - a["prompt_tokens"]) / 200)
    lines = max(1, int(0.98 * (int(sys.argv[3]) - (a["prompt_tokens"] - 100 * per)) / per))
    w1, u = ask(doc(lines, 0), 16)
    w2, _ = ask(doc(lines, 1), 16)
    print(f"{w1:.3f} {w2:.3f} {u.get('prompt_tokens', 0)}")
    sys.exit(0)
s1, _ = ask("Write a Python function that reverses a linked list, with comments.", 16)
s2, _ = ask("Explain in Go how a ring buffer wraps its write index, with a short example.", 16)
l1, _ = ask(doc(180, 0), 16)
l2, _ = ask(doc(180, 1), 16)
print(f"{s1:.3f} {s2:.3f} {l1:.3f} {l2:.3f}")
PY
    srv_stop "$SRV"; SRV=""
}

# slower <first> <second>: the first of a twin pair took longer than its twin allows (x1.25 + 0.25 s): it compiled.
slower() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > 1.25 * b + 0.25) }'; }
# penalty <first> <second>: how much longer the first took (s).
penalty() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", a - b }'; }

FAIL=0
fail() { echo -e "${RED}FAIL${NC} $1"; FAIL=1; }

# ── part 1: every load warms, by default when idle and with --warmup-full before ready ──
if [ -n "$MODEL" ]; then
    echo "  part 1 on $MODEL"
    COPY=$(copy_of "$BINARY" idle); FULL=$(copy_of "$BINARY" full); CONTROL=$(copy_of "$BINARY" control)
    read -r I1 I2 IL1 IL2 <<< "$(timed "$COPY" "$MODEL" "$SCRATCH/idle.log" empty idle short "" $P1_FLAGS)"
    read -r A1 A2 AL1 AL2 <<< "$(timed "$COPY" "$MODEL" "$SCRATCH/idle2.log" keep idle short "" $P1_FLAGS)"
    read -r F1 F2 FL1 FL2 <<< "$(timed "$FULL" "$MODEL" "$SCRATCH/full.log" empty ready short "" --warmup-full $P1_FLAGS)"
    read -r G1 G2 GL1 GL2 <<< "$(timed "$FULL" "$MODEL" "$SCRATCH/full2.log" keep ready short "" --warmup-full $P1_FLAGS)"
    read -r C1 C2 CL1 CL2 <<< "$(timed "$CONTROL" "$MODEL" "$SCRATCH/cold.log" empty ready short "" --no-warmup-eager $P1_FLAGS)"
    [ -n "$IL2" ] && [ -n "$AL2" ] && [ -n "$FL2" ] && [ -n "$GL2" ] && [ -n "$CL2" ] || { echo -e "${RED}FAIL${NC} an arm gave no reading"; exit 1; }
    echo "  twin pairs, wall s (first second), short then long:"
    echo "    default, after its idle warm-up $I1 $I2 / $IL1 $IL2; its second load $A1 $A2 / $AL1 $AL2"
    echo "    --warmup-full $F1 $F2 / $FL1 $FL2; its second load $G1 $G2 / $GL1 $GL2; --no-warmup-eager $C1 $C2 / $CL1 $CL2"
    for log in idle idle2; do
        [ "$(line_of "$READY" "$SCRATCH/$log.log")" -lt "$(line_of "$DONE" "$SCRATCH/$log.log")" ] || fail "$log: the default load did not serve its warm requests after it reported ready"
    done
    for log in full full2; do
        d=$(line_of "$DONE" "$SCRATCH/$log.log")
        [ -n "$d" ] && [ "$d" -lt "$(line_of "$READY" "$SCRATCH/$log.log")" ] || fail "$log: --warmup-full reported ready before it had served its warm requests"
    done
    grep -a -q -F "$DONE" "$SCRATCH/cold.log" && fail "cold: --no-warmup-eager served warm requests"
    # The control proves the reading: unwarmed, the first of a pair compiles and takes visibly longer.
    awk -v a="$C1" -v b="$C2" -v c="$CL1" -v d="$CL2" 'BEGIN { exit !((a - b) + (c - d) >= 0.5) }' || fail "the control's first requests took no longer than their twins ($C1 $C2 / $CL1 $CL2): the reading is blind"
    # Default: the idle plan is short prompts, so its short pair is held to it; --warmup-full: both pairs.
    slower "$I1" "$I2" && fail "after the idle warm-up the first short request took $I1 s against its twin's $I2"
    slower "$A1" "$A2" && fail "the default's second load: the first short request took $A1 s against its twin's $A2"
    for x in "F1 F2 FL1 FL2:$F1 $F2 $FL1 $FL2" "full2:$G1 $G2 $GL1 $GL2"; do
        read -r a b c d <<< "${x#*:}"
        { slower "$a" "$b" || slower "$c" "$d"; } && fail "a --warmup-full load's first requests took longer than their twins ($a $b / $c $d)"
    done
fi

# ── part 2: DeepSeek-V4.1, a load after another under other kernel arms; the widest pass ──
# case_xy <bin> <name> <X env> <X pack> <Y env> <Y pack> [Y flag ...]: load X (--warmup-full, its requests), then Y.
# Y passes when it served its own warm requests and its first requests are no slower than their twins. A load that
# skipped its warm-up because of X compiles its own kernels inside its first requests: they are slower.
DS_FLAGS=${WARM_TEST_DSV41_FLAGS:---ctx-size 32768}
DS_RESULTS="$SCRATCH/ds-results.txt"
: > "$DS_RESULTS"
# PACK_X / PACK_Y: the pack `repoint` puts under one path before X's load and before Y's.
case_xy() {
    local bin="$1" name="$2" xenv="$3" xpack="$4" yenv="$5" ypack="$6"; shift 6
    local yflags="$*" s1 s2 l1 l2
    [ -n "${PACK_X:-}" ] && repoint "$PACK_X"
    timed "$bin" "$xpack" "$SCRATCH/$name-x.log" empty ready short "$xenv" $DS_FLAGS --warmup-full ${XFLAGS:-} >/dev/null
    [ -n "${PACK_Y:-}" ] && repoint "$PACK_Y"
    read -r s1 s2 l1 l2 <<< "$(timed "$bin" "$ypack" "$SCRATCH/$name-y.log" keep ready short "$yenv" $DS_FLAGS --warmup-full $yflags)"
    local served=no; grep -a -q -F "$DONE" "$SCRATCH/$name-y.log" && served=yes
    local verdict=PASS
    { [ "$served" = yes ] && [ -n "$l2" ] && ! slower "$s1" "$s2" && ! slower "$l1" "$l2"; } || verdict=FAIL
    echo "$name: Y served its warm requests $served; Y's twin pairs (s) $s1 $s2 / $l1 $l2, first-request penalty $(penalty "$s1" "$s2") / $(penalty "$l1" "$l2"): $verdict" >> "$DS_RESULTS"
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
        [ -f "$WARM_TEST_DSV41_ALT/config.json" ] || { fail "WARM_TEST_DSV41_ALT $WARM_TEST_DSV41_ALT holds no pack"; return; }
        v=$(PACK_X="$DSV41" PACK_Y="$WARM_TEST_DSV41_ALT" case_xy "$bin" "$tag-pack" "" "$SAMEPATH" "" "$SAMEPATH"); [ "$v" = PASS ] || DS_FAILED+=("$tag replaced-pack")
    fi
}

if [ -n "$DSV41" ]; then
    echo "  part 2 on $DSV41 ($DS_FLAGS)"
    DS_FAILED=()
    NEW=$(copy_of "$BINARY" ds41)
    ds_cases "$NEW" new
    NEW_FAILED=("${DS_FAILED[@]+"${DS_FAILED[@]}"}")
    # The widest pass: after --warmup-full a pair just under the load's widest pass (the width its loads named).
    PASS_ROWS=$(grep -a -o "widest pass [0-9]* rows" "$SCRATCH/new-dspark-y.log" | head -1 | grep -o "[0-9]*")
    [ -n "$PASS_ROWS" ] && [ "$PASS_ROWS" -gt 0 ] || fail "the --warmup-full load named no widest pass"
    WIDE_TOKENS=$(( ${PASS_ROWS:-0} - 600 ))
    read -r W1 W2 WT <<< "$(WIDE_TOKENS=$WIDE_TOKENS timed "$NEW" "$DSV41" "$SCRATCH/wide.log" empty ready wide "" $DS_FLAGS --warmup-full)"
    read -r WC1 WC2 _ <<< "$(WIDE_TOKENS=$WIDE_TOKENS timed "$NEW" "$DSV41" "$SCRATCH/wide-cold.log" empty ready wide "" $DS_FLAGS --no-warmup-eager)"
    echo "    widest pass $PASS_ROWS rows, a $WT-token twin pair (s): after --warmup-full $W1 $W2, unwarmed $WC1 $WC2"
    awk -v a="$WC1" -v b="$WC2" 'BEGIN { exit !(a - b >= 0.5) }' || fail "the unwarmed widest prompt took no longer than its twin ($WC1 $WC2): the reading is blind"
    slower "${W1:-0}" "${W2:-0}" && fail "after --warmup-full the widest-pass prompt took $W1 s against its twin's $W2"
    for f in "${NEW_FAILED[@]+"${NEW_FAILED[@]}"}"; do fail "a DeepSeek-V4.1 load after another compiled its own kernels in its first requests: $f"; done
    if [ -n "${WARM_TEST_OLD_BINARY:-}" ]; then
        DS_FAILED=()
        OLD=$(copy_of "$WARM_TEST_OLD_BINARY" old)
        ds_cases "$OLD" old
        read -r OW1 OW2 _ <<< "$(WIDE_TOKENS=$WIDE_TOKENS timed "$OLD" "$DSV41" "$SCRATCH/old-wide.log" empty ready wide "" $DS_FLAGS --warmup-full)"
        echo "    the other build ($WARM_TEST_OLD_BINARY): fails ${DS_FAILED[*]:-none}; its widest-pass pair after --warmup-full $OW1 $OW2"
    fi
    sed 's/^/    /' "$DS_RESULTS"
fi
[ "$FAIL" -eq 0 ] && echo -e "${GREEN}PASS${NC} every load warms its own kernels (when idle, or before ready with --warmup-full), whatever an earlier load compiled"
exit "$FAIL"
