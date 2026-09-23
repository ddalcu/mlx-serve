#!/usr/bin/env bash
# Laya typed decisions end-to-end over HTTP (`POST /v1/decisions`).
#
#   LAYA_MODEL=<checkpoint dir> ./tests/test_laya_decisions.sh [port]
#
# Defaults to the HF cache snapshot of aac6fef/laya-multilingual-mlx. The
# server is started with `--model-dir <root>` where the checkpoint is exposed
# as <root>/aac6fef/laya-multilingual-mlx, so discovery + on-demand load are
# exercised, not just the forward. Answers are compared against the laya_mlx
# fixtures in tests/fixtures/laya/cases.json (tolerance 0.01), then the
# request latency (3 questions) is measured as the median of 30.
#
# Hermetic counterparts: the unit tests in src/laya.zig (LAYA_TEST_MODEL +
# LAYA_FIXTURES) pin token ids, hidden states and probabilities.
set -uo pipefail

PORT="${1:-11441}"
HF_SNAP="$(command ls -d "$HOME"/.cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/*/ 2>/dev/null | head -1)"
MODEL="${LAYA_MODEL:-${HF_SNAP%/}}"
BIN="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
FIXTURES="tests/fixtures/laya/cases.json"
LOG="/tmp/laya-test-$PORT.log"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT

if [ ! -f "$MODEL/rl_agent_config.json" ]; then echo "SKIP: no Laya checkpoint at '$MODEL' (set LAYA_MODEL)"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; fi
if [ ! -f "$FIXTURES" ]; then echo "FAIL: missing $FIXTURES (tests/dump_laya_fixtures.py)"; exit 1; fi

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

# Two-level org/name root, like ~/.mlx-serve/models.
ROOT="$TMP/models"
mkdir -p "$ROOT/aac6fef"
ln -s "$MODEL" "$ROOT/aac6fef/laya-multilingual-mlx"
MODEL_ID="aac6fef/laya-multilingual-mlx"

"$BIN" --serve --port "$PORT" --model-dir "$ROOT" --log-level info > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do curl -s -f "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s -f "localhost:$PORT/health" >/dev/null || { echo "FAIL: server not healthy"; tail -20 "$LOG"; exit 1; }

echo "=== discovery ==="
MODELS="$(curl -s "localhost:$PORT/v1/models")"
check "model discovered from --model-dir" "$(echo "$MODELS" | python3 -c "import sys,json; print(any(m['id']=='$MODEL_ID' for m in json.load(sys.stdin)['data']))")" "True"
check "stub advertises the decisions capability" "$(echo "$MODELS" | python3 -c "import sys,json; print('decisions' in next(m for m in json.load(sys.stdin)['data'] if m['id']=='$MODEL_ID').get('capabilities',[]))")" "True"

decide() { curl -s -m 300 -X POST "localhost:$PORT/v1/decisions" -H 'content-type: application/json' -d "$1"; }
expect_code() { # label, expected, body, [path]
  local path="${4:-/v1/decisions}" got
  got="$(curl -s -o "$TMP/err.json" -w '%{http_code}' -m 300 -X POST "localhost:$PORT$path" -H 'content-type: application/json' -d "$3")"
  check "$1" "$got" "$2"
}

echo "=== parity vs laya_mlx fixtures (tolerance 0.01) ==="
python3 - "$FIXTURES" "$MODEL_ID" > "$TMP/bodies.txt" <<'EOF'
import json, sys
fx = json.load(open(sys.argv[1]))
for lang, state in fx["states"].items():
    print(lang + "\t" + json.dumps({"model": sys.argv[2], "state": state, "questions": fx["questions"]}, ensure_ascii=False))
EOF
while IFS=$'\t' read -r lang body; do
  decide "$body" > "$TMP/out_$lang.json"
  RES="$(python3 - "$FIXTURES" "$TMP/out_$lang.json" "$lang" <<'EOF'
import json, sys
fx = json.load(open(sys.argv[1])); got = json.load(open(sys.argv[2])); lang = sys.argv[3]
if "answers" not in got: print("no answers: " + json.dumps(got)[:200]); sys.exit()
worst = 0.0; problems = []
for c in fx["cases"]:
    if c["lang"] != lang: continue
    want = c["expected"]; have = got["answers"].get(c["qid"])
    if have is None: problems.append(c["qid"] + " missing"); continue
    for k, v in want.items():
        h = have.get(k)
        if isinstance(v, str):
            if h != v: problems.append(f"{c['qid']}.{k}: {h!r} != {v!r}")
        elif isinstance(v, (int, float)):
            d = abs(h - v); worst = max(worst, d)
            if d > 0.01: problems.append(f"{c['qid']}.{k}: {h} vs {v}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                hh = h.get(kk)
                if isinstance(vv, str):
                    if hh != vv: problems.append(f"{c['qid']}.{k}.{kk}: {hh!r} != {vv!r}")
                else:
                    d = abs(hh - vv); worst = max(worst, d)
                    if d > 0.01: problems.append(f"{c['qid']}.{k}.{kk}: {hh} vs {vv}")
expected_tokens = sum(len(c["ids"]) for c in fx["cases"] if c["lang"] == lang)
if got["usage"]["input_tokens"] != expected_tokens: problems.append(f"usage.input_tokens {got['usage']['input_tokens']} != {expected_tokens}")
print(("ok" if not problems else "; ".join(problems)) + f" max|diff|={worst:.4f}")
EOF
)"
  case "$RES" in ok*) ok "$lang: $RES";; *) bad "$lang: $RES";; esac
done < "$TMP/bodies.txt"

echo "=== request validation ==="
expect_code "missing questions -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\"}"
expect_code "unknown question type -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"rank\",\"instructions\":\"?\"}}}"
expect_code "choice without criteria -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"?\"}}}"
expect_code "invalid JSON -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":"
expect_code "chat on a decision model -> 400" 400 "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" /v1/chat/completions
check "chat refusal names /v1/decisions" "$(grep -c '/v1/decisions' "$TMP/err.json")" "1"
expect_code "string state + list criteria + noul criteria -> 200" 200 "{\"model\":\"$MODEL_ID\",\"state\":\"Refund me now or I cancel.\",\"questions\":{\"team\":{\"type\":\"choice\",\"instructions\":\"Which team?\",\"criteria\":[\"billing\",\"sales\"]},\"churn\":{\"type\":\"noul\",\"instructions\":\"Threatens to cancel?\",\"criteria\":{\"false\":\"no threat\",\"true\":\"explicit threat\"}}}}"
check "loaded model advertises decisions" "$(curl -s "localhost:$PORT/v1/models" | python3 -c "import sys,json; print('decisions' in next(m for m in json.load(sys.stdin)['data'] if m['id']=='$MODEL_ID').get('capabilities',[]))")" "True"

echo "=== mixed-length questions: same answers as one at a time, in request order ==="
MIX="$(python3 - "$PORT" "$MODEL_ID" <<'EOF'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
def post(qs):
    body = json.dumps({"model": model, "state": "I was charged twice for my order and want a refund today", "questions": qs}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/decisions", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["answers"]
qs = {f"q{i}": {"type": "choice", "instructions": "Which team? " + "Read every detail. " * (i * 7 % 40), "criteria": ["billing", "sales", "tech"]} for i in range(20)}
together = post(qs)
diff = max(abs(together[k]["probabilities"][l] - post({k: v})[k]["probabilities"][l]) for k, v in qs.items() for l in ("billing", "sales", "tech"))
print("same" if list(together) == list(qs) and diff <= 1e-3 else f"differ {diff} {list(together)}")
EOF
)"
check "20 mixed-length questions match their one-at-a-time answers, in request order" "$MIX" "same"

echo "=== latency: 1 request, 3 questions (en state), median of 30 after 5 warm-ups ==="
BODY="$(grep '^en	' "$TMP/bodies.txt" | cut -f2-)"
for _ in 1 2 3 4 5; do decide "$BODY" >/dev/null; done
python3 - "$PORT" "$BODY" <<'EOF'
import json, sys, time, urllib.request, statistics
port, body = sys.argv[1], sys.argv[2].encode()
times = []
for _ in range(30):
    req = urllib.request.Request(f"http://localhost:{port}/v1/decisions", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter(); urllib.request.urlopen(req).read(); times.append((time.perf_counter() - t) * 1000)
times.sort()
print(f"  median {statistics.median(times):.1f} ms  p10 {times[2]:.1f} ms  p90 {times[26]:.1f} ms  (HTTP round trip incl. tokenization)")
EOF
grep -h '\[decision\] .* ms' "$LOG" | tail -3 | sed 's/^/  server: /' || true

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
