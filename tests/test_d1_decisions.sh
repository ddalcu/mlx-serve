#!/usr/bin/env bash
# D1 typed decisions end-to-end over HTTP (`POST /v1/decisions`).
#
#   D1_MODEL=<bf16 D1 dir> ./tests/test_d1_decisions.sh [port]
#
# The checkpoint is LiquidAI/d1-3B in its bf16 layout (config.json names modeling_d1.D1Model). The server starts
# with `--model-dir <root>` where the pack is exposed as <root>/LiquidAI/d1-3B, so discovery (the auto_map marker on
# an LFM2-VL config) and the on-demand load are exercised, not just the forward. Answers are compared against the
# card's own package on CPU in float32 (tests/fixtures/d1/cases.json, from tests/dump_d1_fixtures.py): bf16 Metal
# agrees to 0.02 on probabilities, and input token counts must match exactly.
# Hermetic counterparts: the unit tests in src/plugins/d1/d1.zig and the D1 discovery test in src/host/registry/discovery.zig.
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11445}"
MODEL="${D1_MODEL:-$HOME/.mlx-serve/models/LiquidAI/d1-3B}"
BIN="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
FIXTURES="tests/fixtures/d1/cases.json"
LOG="/tmp/d1-test-$PORT.log"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT

if ! grep -q modeling_d1.D1Model "$MODEL/config.json" 2>/dev/null; then echo "SKIP: no D1 checkpoint at '$MODEL' (set D1_MODEL)"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; fi
if [ ! -f "$FIXTURES" ]; then echo "FAIL: missing $FIXTURES (tests/dump_d1_fixtures.py)"; exit 1; fi

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

ROOT="$TMP/models"
mkdir -p "$ROOT/LiquidAI"
ln -s "$MODEL" "$ROOT/LiquidAI/d1-3B"
MODEL_ID="LiquidAI/d1-3B"

"$BIN" --serve --port "$PORT" --model-dir "$ROOT" --log-level info > "$LOG" 2>&1 &
SRV=$!
srv_wait_health "localhost:$PORT" 60 1 || { echo "FAIL: server not healthy"; tail -20 "$LOG"; exit 1; }

echo "=== discovery ==="
MODELS="$(curl -s "localhost:$PORT/v1/models")"
row() { echo "$MODELS" | python3 -c "import sys,json; m=next(m for m in json.load(sys.stdin)['data'] if m['id']=='$MODEL_ID'); print($1)"; }
check "pack discovered from --model-dir" "$(echo "$MODELS" | python3 -c "import sys,json; print(any(m['id']=='$MODEL_ID' for m in json.load(sys.stdin)['data']))")" "True"
check "advertises decisions and nothing else" "$(row "m.get('capabilities')")" "['decisions']"

decide() { curl -s -m 300 -X POST "localhost:$PORT/v1/decisions" -H 'content-type: application/json' -d "$1"; }
expect_code() { # label, expected, body, [path]
  local path="${4:-/v1/decisions}" got
  got="$(curl -s -o "$TMP/err.json" -w '%{http_code}' -m 300 -X POST "localhost:$PORT$path" -H 'content-type: application/json' -d "$3")"
  check "$1" "$got" "$2"
}

cat > "$TMP/parity.py" <<'PYEOF'
import copy, json, sys, urllib.request

fx, port, model = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
bad, n = [], 0

def post(body):
    req = urllib.request.Request(f"http://localhost:{port}/v1/decisions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))

def compare(case, got, label):
    global n
    if got["usage"]["input_tokens"] != case["input_tokens"]:
        bad.append(f"{label}: input_tokens {got['usage']['input_tokens']} != {case['input_tokens']}")
    for qid, want in case["answers"].items():
        have = got["answers"][qid]
        n += 1
        if have["type"] != want["type"]:
            bad.append(f"{label}/{qid}: type {have['type']} != {want['type']}")
            continue
        if want["type"] == "noul":
            diff = abs(have["noul"] - want["noul"])
        else:
            if want["type"] == "choice" and have["choice"] != want["choice"]:
                bad.append(f"{label}/{qid}: choice {have['choice']} != {want['choice']}")
            diff = max(abs(have["probabilities"][k] - v) for k, v in want["probabilities"].items())
            if want["type"] == "score":
                diff = max(diff, abs(have["score"] - want["score"]))
        if diff > 0.02:
            bad.append(f"{label}/{qid}: off by {diff:.4f}")

for case in fx["cases"]:
    compare(case, post({"model": model, "state": case["state"], "questions": case["questions"]}), case["name"])

# The app sends choice criteria as a plain list of labels; the card's own form is a map with null descriptions.
case = next(c for c in fx["cases"] if c["name"] == "demo_default")
listed = copy.deepcopy(case["questions"])
listed["team"]["criteria"] = list(listed["team"]["criteria"].keys())
compare(case, post({"model": model, "state": case["state"], "questions": listed}), "demo_default(list)")

print(f"{n} answers, {len(bad)} off")
for b in bad:
    print("  " + b)
sys.exit(1 if bad else 0)
PYEOF

echo "=== parity vs the card's own package (float32 CPU; probabilities within 0.02, tokens exact) ==="
PAR="$(python3 -I "$TMP/parity.py" "$FIXTURES" "$PORT" "$MODEL_ID" 2>&1)"
PAR_EXIT=$?
echo "$PAR" | sed 's/^/    /'
check "every answer matches the card's package" "$PAR_EXIT" "0"
check "the decision engine is the D1 one" "$(grep -c '\[decision\] d1 engine ready' "$LOG")" "1"

echo "=== request contract ==="
DEMO='{"model":"'"$MODEL_ID"'","state":"Refund me now.","questions":{"churn":{"type":"noul","instructions":"Cancel?"}}}'
expect_code "a picture is refused by name (text-only build)" 400 '{"model":"'"$MODEL_ID"'","state":"x","questions":{"churn":{"type":"noul","instructions":"Cancel?"}},"images":["data:image/png;base64,AAAA"]}'
grep -q "text-only" "$TMP/err.json" && ok "the 400 names the text-only build" || bad "the 400 names the text-only build"
expect_code "a one-level score is refused" 400 '{"model":"'"$MODEL_ID"'","state":"x","questions":{"u":{"type":"score","instructions":"x","criteria":["one"]}}}'
expect_code "an unknown question type is refused" 400 '{"model":"'"$MODEL_ID"'","state":"x","questions":{"u":{"type":"rank","instructions":"x"}}}'
expect_code "a choice without criteria is refused" 400 '{"model":"'"$MODEL_ID"'","state":"x","questions":{"u":{"type":"choice","instructions":"x"}}}'
expect_code "chat is refused and points at /v1/decisions" 400 '{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"hi"}]}' /v1/chat/completions
check "a decision request answers 200" "$(curl -s -o /dev/null -w '%{http_code}' -m 300 -X POST "localhost:$PORT/v1/decisions" -H 'content-type: application/json' -d "$DEMO")" "200"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
