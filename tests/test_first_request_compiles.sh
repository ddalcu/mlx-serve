#!/bin/bash
# test_first_request_compiles.sh — the first time a build loads a model it warms its Metal kernels with requests of
# its own, so the prompts someone sends add next to nothing to Metal's shader cache; a later load owes nothing and
# serves none. By default the model is ready first and warms when idle (short prompts only); with --warmup-full it
# warms before it reports ready, long prompts included. The server runs as a copy of the binary under its own bundle
# id, so its shader cache starts EMPTY (a build's first run on a machine), and the cache index is sized around a short
# chat, a short prompt and another question under it, then a long prompt and another question under that. A copy with
# --no-warmup-eager is the control: there the first prompts compile what the load did not, and the test must see it.
#
#   ./tests/test_first_request_compiles.sh [port]   # macOS only. BINARY, WARM_TEST_MODEL=<org>/<repo>
#
# Hermetic twins: `promptIds`, the plans' forks and ladder, `owed`, `nextStep`, the idle warm-up's tests
# (warm_requests.zig), `ramTwin`, `submitNow`.

set -u
source "$(dirname "$0")/lib/server.sh"
source "$(dirname "$0")/_lib_models.sh"

PORT="${1:-11391}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

[ "$(uname)" = Darwin ] || { echo -e "${YELLOW}SKIP${NC} first-request compiles: Metal only."; exit 0; }
[ -x "$BINARY" ] || { echo -e "${RED}FAIL${NC} no $BINARY: build ReleaseFast first"; exit 1; }
MODEL=$(find_fitting_model ${WARM_TEST_MODEL:-mlx-community/gemma-4-e4b-it-4bit ddalcu/Qwen3.8-27B-MLX-Serve-4bit}) || {
    echo -e "${YELLOW}SKIP${NC} first-request compiles: no test model on this box."; exit 0; }
echo "  using $MODEL"

# Metal keys its on-disk cache on the bundle id every build embeds: a copy under another id of the same length
# starts with none.
ID_OLD=com.dalcu.mlx-core.mlx-serve
ID_NEW=com.dalcu.mlx-cold.mlx-serve
COPY="$(dirname "$BINARY")/mlx-serve-coldcache.$$"
FULL="$COPY.full"
CONTROL="$COPY.control"
CACHE="$(getconf DARWIN_USER_CACHE_DIR)$ID_NEW"
MARKERS="$(getconf DARWIN_USER_CACHE_DIR)mlx-serve-warm"
SCRATCH="$HOME/claude-tmp/test-first-request-$$"
SRV=""
mkdir -p "$SCRATCH" "$MARKERS"
ls "$MARKERS" > "$SCRATCH/markers.before"
cleanup() {
    srv_stop "$SRV"
    # Only the markers this run wrote: the others are real builds' on this machine.
    ls "$MARKERS" 2>/dev/null | grep -vxF -f "$SCRATCH/markers.before" | while read -r m; do rm -f "$MARKERS/$m"; done
    rm -rf "$COPY" "$FULL" "$CONTROL" "$CACHE" "$SCRATCH"
}
trap cleanup EXIT
# A marker names the build and where it runs from, so each copy is a build that has loaded nothing yet.
for dst in "$COPY" "$FULL" "$CONTROL"; do
    python3 - "$BINARY" "$dst" "$ID_OLD" "$ID_NEW" <<'PY' || { echo -e "${RED}FAIL${NC} the binary does not embed its bundle id once"; exit 1; }
import os, sys
src, dst, old, new = sys.argv[1], sys.argv[2], sys.argv[3].encode(), sys.argv[4].encode()
data = open(src, "rb").read()
assert data.count(old) == 1 and len(old) == len(new)
open(dst, "wb").write(data.replace(old, new))
os.chmod(dst, 0o755)
PY
    codesign -f -s - "$dst" 2>/dev/null
done

READY="Model ready (loaded on inference thread)"
DONE="Warm requests complete"
line_of() { grep -a -n -m1 -F "$1" "$2" | cut -d: -f1; }

# compiled <binary> <log> <keep|empty> <idle|ready> [flag]: bytes the shader-library index gained over the short
# prompts, then over the long ones, the shader cache emptied first or kept as the last boot left it. `idle` waits for
# the idle warm-up to end before the first prompt; `ready` sends it as soon as /health answers.
compiled() {
    local bin="$1" log="$2" cache="$3" start="$4"; shift 4
    [ "$cache" = empty ] && rm -rf "$CACHE"
    rm -rf "$SCRATCH/home"; mkdir -p "$SCRATCH/home"
    HOME="$SCRATCH/home" "$bin" --serve --model "$MODEL" --port "$PORT" --host 127.0.0.1 "$@" >"$log" 2>&1 &
    SRV=$!
    srv_wait_health "$BASE" 600 0.5 "$SRV" || { echo -e "${RED}FAIL${NC} the server never came up" >&2; tail -5 "$log" >&2; srv_stop "$SRV"; exit 1; }
    if [ "$start" = idle ]; then
        for _ in $(seq 1 600); do grep -a -q -F "$DONE" "$log" && break; sleep 0.5; done
        grep -a -q -F "$DONE" "$log" || { echo -e "${RED}FAIL${NC} the idle warm-up never ended" >&2; tail -5 "$log" >&2; srv_stop "$SRV"; exit 1; }
    fi
    python3 - "$BASE" "$CACHE" <<'PY' || { echo -e "${RED}FAIL${NC} a request failed, or the first one found a prompt cached" >&2; srv_stop "$SRV"; exit 1; }
import glob, json, os, sys, urllib.request
def index():
    return sum(os.path.getsize(p) for p in glob.glob(sys.argv[2] + "/**/libraries.list", recursive=True))
def ask(text, n):
    body = dict(model="x", messages=[{"role": "user", "content": text}], max_tokens=n, temperature=0)
    req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["usage"]
def doc(lines):
    return " ".join(f"[{i}] The archive records the following entry." for i in range(lines))
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

read -r IDLE_S IDLE_L <<< "$(compiled "$COPY" "$SCRATCH/idle.log" empty idle)"
read -r AGAIN_S AGAIN_L <<< "$(compiled "$COPY" "$SCRATCH/idle2.log" keep ready)"
read -r FULL_S FULL_L <<< "$(compiled "$FULL" "$SCRATCH/full.log" empty ready --warmup-full)"
# The full plan holds the minimal one: without the flag this copy owes nothing either.
read -r FULL2_S FULL2_L <<< "$(compiled "$FULL" "$SCRATCH/full2.log" keep ready)"
read -r COLD_S COLD_L <<< "$(compiled "$CONTROL" "$SCRATCH/cold.log" empty ready --no-warmup-eager)"
[ -n "$IDLE_L" ] && [ -n "$AGAIN_L" ] && [ -n "$FULL_L" ] && [ -n "$FULL2_L" ] && [ -n "$COLD_L" ] || { echo -e "${RED}FAIL${NC} an arm gave no reading"; exit 1; }
echo "  shader-library index bytes gained, short prompts + long prompts:"
echo "    default, after its idle warm-up $IDLE_S + $IDLE_L; its second load $AGAIN_S + $AGAIN_L"
echo "    --warmup-full $FULL_S + $FULL_L; its second load $FULL2_S + $FULL2_L; --no-warmup-eager $COLD_S + $COLD_L"

FAIL=0
fail() { echo -e "${RED}FAIL${NC} $1"; FAIL=1; }
# Default: ready first, the warm requests after it. --warmup-full: the warm requests first.
[ "$(line_of "$READY" "$SCRATCH/idle.log")" -lt "$(line_of "$DONE" "$SCRATCH/idle.log")" ] || fail "the default load served its warm requests before it reported ready"
FULL_DONE=$(line_of "$DONE" "$SCRATCH/full.log")
[ -n "$FULL_DONE" ] && [ "$FULL_DONE" -lt "$(line_of "$READY" "$SCRATCH/full.log")" ] || fail "--warmup-full reported ready before it had served its warm requests"
for log in idle2 full2 cold; do
    grep -a -q -F "$DONE" "$SCRATCH/$log.log" && fail "$log: a load that owes no warm requests served them"
done
# The control proves the reading: without the warm requests these prompts compile shaders, and plenty of them.
COLD=$(( COLD_S + COLD_L ))
[ "$COLD_S" -ge 500 ] && [ "$COLD" -ge 1000 ] || fail "the control compiled nearly nothing ($COLD_S + $COLD_L bytes): the reading is blind"
[ $(( IDLE_S * 3 )) -le "$COLD_S" ] || fail "after the idle warm-up the short prompts compiled $IDLE_S bytes of index, more than a third of the unwarmed $COLD_S"
[ $(( AGAIN_S * 3 )) -le "$COLD_S" ] || fail "the default's second load compiled $AGAIN_S bytes of index on the short prompts"
[ $(( (FULL_S + FULL_L) * 3 )) -le "$COLD" ] || fail "after --warmup-full the prompts compiled $(( FULL_S + FULL_L )) bytes of index, more than a third of the unwarmed $COLD"
[ $(( (FULL2_S + FULL2_L) * 3 )) -le "$COLD" ] || fail "the second load after --warmup-full compiled $(( FULL2_S + FULL2_L )) bytes of index"
[ "$FAIL" -eq 0 ] && echo -e "${GREEN}PASS${NC} a build's first load warms its kernels (when idle, or before ready with --warmup-full), and only that load"
exit "$FAIL"
