#!/bin/bash
# test_suppress_cjk.sh — `suppress_cjk` on the three chat surfaces and /v1/completions: without it a
# CJK-provoking prompt writes CJK, with it no reply carries a CJK character; a non-boolean is a 400, null
# leaves it off, and a greedy PLD round cannot copy CJK from the prompt past the mask.
#
#   SUPPRESS_CJK_MODEL=<dir> ./tests/test_suppress_cjk.sh [port]   # default Qwen3.5-4B-MLX-8bit; RUNS=10 per arm
set -uo pipefail
cd "$(dirname "$0")/.."
source tests/lib/server.sh
source tests/_lib_models.sh

MODEL="${SUPPRESS_CJK_MODEL:-$(find_model mlx-community/Qwen3.5-4B-MLX-8bit)}"
PORT="${1:-11514}"
RUNS="${RUNS:-10}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: no Qwen3.5-4B-MLX-8bit (SUPPRESS_CJK_MODEL)"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }
model_fits "$MODEL" || { echo "SKIP: $(model_gb "$MODEL") GB > $(max_model_gb) GB"; exit 0; }

mkdir -p "$HOME/claude-tmp"
WORK=$(mktemp -d "$HOME/claude-tmp/suppress-cjk.XXXXXX")
mkdir -p "$WORK/home"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 0.5
HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --log-level info --max-concurrent 4 > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
trap 'srv_stop "$SERVER_PID"' EXIT
srv_wait_health "$BASE" 180 1 "$SERVER_PID" || { echo "FAIL: server did not start"; tail -20 "$WORK/server.log"; exit 1; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }
post() { curl -s -m 300 -o "$WORK/body" -w '%{http_code}' "$BASE$1" -H 'Content-Type: application/json' --data-binary "$2"; }
# CJK characters anywhere in the reply; the requests below carry none unless the test says so.
cjk() {
    python3 - "$WORK/body" <<'PY'
import json, sys
R = ((0x1100, 0x11FF), (0x2E80, 0xDFFF), (0xF900, 0xFAFF), (0xFE10, 0xFE1F), (0xFE30, 0xFE6F), (0xFF00, 0xFFEF),
     (0x16FE0, 0x16FFF), (0x1AFF0, 0x1B16F), (0x1F200, 0x1F2FF), (0x20000, 0x3FFFF))
def walk(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from walk(x)
    elif isinstance(v, list):
        for x in v:
            yield from walk(x)
print(sum(1 for s in walk(json.load(open(sys.argv[1]))) for c in s if any(lo <= ord(c) <= hi for lo, hi in R)))
PY
}
route() { case "$1" in chat) echo /v1/chat/completions ;; messages) echo /v1/messages ;; responses) echo /v1/responses ;; completions) echo /v1/completions ;; esac; }
body() { # <surface> <prompt> <max tokens> <extra fields>
    case "$1" in
        responses) echo "{\"model\":\"m\",\"max_output_tokens\":$3,\"input\":\"$2\"$4}" ;;
        completions) echo "{\"model\":\"m\",\"max_tokens\":$3,\"prompt\":\"$2\"$4}" ;;
        *)         echo "{\"model\":\"m\",\"max_tokens\":$3,\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]$4}" ;;
    esac
}
Q='Wie heisst die Hauptstadt Chinas in Originalschrift? Antworte nur mit dem Namen.'
# A raw prompt has no chat turn to ask in: a list that continues in Han. The reply does not echo it.
QC='English: Shanghai\nChinese: 上海\nEnglish: Guangzhou\nChinese: 广州\nEnglish: Beijing\nChinese:'

echo "=== CJK with and without the field ($RUNS runs each) ==="
for s in chat messages responses completions; do
    off=0; on=0; errs=0; q=$Q; n=512
    [[ $s == completions ]] && { q=$QC; n=16; }
    for i in $(seq 1 "$RUNS"); do
        [[ "$(post "$(route $s)" "$(body $s "$q" $n ',"temperature":0.7')")" == 200 ]] || errs=$((errs+1))
        [[ "$(cjk)" -gt 0 ]] && off=$((off+1))
        [[ "$(post "$(route $s)" "$(body $s "$q" $n ',"temperature":0.7,"suppress_cjk":true')")" == 200 ]] || errs=$((errs+1))
        [[ "$(cjk)" -gt 0 ]] && on=$((on+1))
    done
    [[ $errs == 0 ]] && ok "$s: all answered" || bad "$s: $errs errors"
    [[ $off -gt 0 ]] && ok "$s: without the field $off/$RUNS write CJK (control)" || bad "$s: control" "no CJK without the field"
    [[ $on == 0 ]] && ok "$s: with the field 0/$RUNS write CJK" || bad "$s: CJK despite the field" "$on/$RUNS runs"
done

echo "=== field validation ==="
for s in chat messages responses completions; do
    code=$(post "$(route $s)" "$(body $s hi 8 ',"suppress_cjk":"true"')")
    [[ $code == 400 ]] && ok "$s: string is a 400" || bad "$s: string" "got $code"
    code=$(post "$(route $s)" "$(body $s hi 8 ',"suppress_cjk":null')")
    [[ $code == 200 ]] && ok "$s: null is unset" || bad "$s: null" "got $code"
done

echo "=== greedy PLD cannot copy CJK from the prompt ==="
P='Wiederhole den folgenden Text exakt, Zeichen für Zeichen, ohne Zusatz: 北京是中国的首都。北京是中国的首都。北京是中国的首都。'
post /v1/chat/completions "$(body chat "$P" 60 ',"temperature":0,"enable_pld":true')" >/dev/null
[[ "$(cjk)" -gt 0 ]] && ok "without the field the copy goes through (control)" || bad "PLD control" "no CJK without the field"
post /v1/chat/completions "$(body chat "$P" 60 ',"temperature":0,"enable_pld":true,"suppress_cjk":true')" >/dev/null
[[ "$(cjk)" == 0 ]] && ok "with the field no CJK under greedy PLD" || bad "CJK through greedy PLD" "$(head -c 160 "$WORK/body")"

grep -q '\[suppress\] suppress_cjk ready' "$WORK/server.log" && ok "engagement line logged" || bad "no [suppress] suppress_cjk ready line"
curl -sf -m 2 "$BASE/health" >/dev/null && ok "server alive" || bad "server died"

echo "PASS=$PASS FAIL=$FAIL"
[[ $FAIL == 0 ]]
