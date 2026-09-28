#!/usr/bin/env bash
# Kev typed decisions end-to-end over HTTP (`POST /v1/decisions`).
#
#   KEV_MODEL=<pack dir> ./tests/test_kev_decisions.sh [port]
#
# A pack is what tests/convert_kev_weights.py writes (stock jaredpalmer/kev-4b; 8-bit by default). The server
# starts with `--model-dir <root>` where the pack is exposed as <root>/jaredpalmer/Kev-4B, so discovery (the pack's
# kev_config.json must win over its qwen3_5 config.json) and on-demand load are exercised, not just the forward.
# Answers are compared against kev's own MLX backend (tests/fixtures/kev/cases.json, tolerance 0.01).
#
# Hermetic counterparts: the unit tests in src/kev.zig (KEV_TEST_MODEL + KEV_FIXTURES pin token ids and
# probabilities) and the Kev discovery test in src/model_discovery.zig.
set -uo pipefail

PORT="${1:-11443}"
MODEL="${KEV_MODEL:-$HOME/.mlx-serve/models/jaredpalmer/Kev-4B-MLX-Serve-8bit}"
BIN="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
FIXTURES="tests/fixtures/kev/cases.json"
LOG="/tmp/kev-test-$PORT.log"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT

if [ ! -f "$MODEL/kev_config.json" ]; then echo "SKIP: no Kev pack at '$MODEL' (set KEV_MODEL)"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; fi
if [ ! -f "$FIXTURES" ]; then echo "FAIL: missing $FIXTURES (tests/dump_kev_fixtures.py)"; exit 1; fi

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

ROOT="$TMP/models"
mkdir -p "$ROOT/jaredpalmer"
ln -s "$MODEL" "$ROOT/jaredpalmer/Kev-4B"
MODEL_ID="jaredpalmer/Kev-4B"

"$BIN" --serve --port "$PORT" --model-dir "$ROOT" --log-level info > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do curl -s -f "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s -f "localhost:$PORT/health" >/dev/null || { echo "FAIL: server not healthy"; tail -20 "$LOG"; exit 1; }

echo "=== discovery ==="
MODELS="$(curl -s "localhost:$PORT/v1/models")"
row() { echo "$MODELS" | python3 -c "import sys,json; m=next(m for m in json.load(sys.stdin)['data'] if m['id']=='$MODEL_ID'); print($1)"; }
check "pack discovered from --model-dir" "$(echo "$MODELS" | python3 -c "import sys,json; print(any(m['id']=='$MODEL_ID' for m in json.load(sys.stdin)['data']))")" "True"
check "classified as kev, not its qwen3_5 base" "$(row "(m.get('meta') or {}).get('architecture')")" "kev"
check "advertises decisions and nothing else" "$(row "m.get('capabilities')")" "['decisions']"

decide() { curl -s -m 300 -X POST "localhost:$PORT/v1/decisions" -H 'content-type: application/json' -d "$1"; }
expect_code() { # label, expected, body, [path]
  local path="${4:-/v1/decisions}" got
  got="$(curl -s -o "$TMP/err.json" -w '%{http_code}' -m 300 -X POST "localhost:$PORT$path" -H 'content-type: application/json' -d "$3")"
  check "$1" "$got" "$2"
}

echo "=== parity vs kev's MLX backend (tolerance 0.01) ==="
PAR="$(python3 - "$FIXTURES" "$PORT" "$MODEL_ID" <<'EOF'
import json, sys, urllib.request
fx, port, model = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
out = []
for c in fx["cases"]:
    body = dict(c["request"], model=model)
    got = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/decisions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    worst, problems = 0.0, []
    for qid, want in c["answers"].items():
        have = got["answers"][qid]
        for k, v in want.items():
            if isinstance(v, str):
                if have.get(k) != v: problems.append(f"{qid}.{k} {have.get(k)!r} != {v!r}")
            elif isinstance(v, (int, float)):
                worst = max(worst, abs(have[k] - v))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str):
                        if have[k].get(kk) != vv: problems.append(f"{qid}.{k}.{kk}")
                    else:
                        worst = max(worst, abs(have[k][kk] - vv))
    if worst > 0.01: problems.append(f"max|diff| {worst:.4f}")
    if got["usage"]["input_tokens"] != len(c["ids"]): problems.append(f"input_tokens {got['usage']['input_tokens']} != {len(c['ids'])}")
    out.append(c["name"] + ("" if not problems else ": " + "; ".join(problems)))
print(f"{len(out)} cases: " + " | ".join(out))
sys.exit(1 if not out or any(":" in o for o in out) else 0)
EOF
)"
RC=$?
if [ "$RC" -eq 0 ] && [ -n "$PAR" ]; then ok "fixture parity within 0.01, token counts equal: $PAR"; else bad "fixture parity (exit $RC): $PAR"; fi

echo "=== request validation (Kev's rules, named 400s) ==="
expect_code "missing questions -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\"}"
expect_code "empty questions -> 400 (kev requires at least one)" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{}}"
expect_code "unknown question type -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"rank\"}}}"
expect_code "choice criteria with a repeated label -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":[\"a\",\"a\"]}}}"
check "  ... and the message says what to send" "$(grep -c 'list of unique labels' "$TMP/err.json")" "1"
expect_code "score with no levels -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[]}}}"
expect_code "noul criteria as a list -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\",\"criteria\":[1]}}}"
expect_code "lone surrogate in the state -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"a\\ud800b\",\"questions\":{\"q\":{\"type\":\"noul\"}}}"
expect_code "lone surrogate in instructions -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"\\udc00\"}}}"
expect_code "invalid JSON -> 400" 400 "{\"model\":\"$MODEL_ID\",\"state\":"
expect_code "chat on a Kev pack -> 400" 400 "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" /v1/chat/completions
check "  ... and the refusal names /v1/decisions" "$(grep -c '/v1/decisions' "$TMP/err.json")" "1"
expect_code "instructions omitted -> 200 (kev renders them empty)" 200 "{\"model\":\"$MODEL_ID\",\"state\":\"Refund me now or I cancel.\",\"questions\":{\"churn\":{\"type\":\"noul\",\"criteria\":{\"true\":\"explicit threat\"}}}}"

echo "=== limits: refused before any model work ==="
python3 - "$TMP" "$MODEL_ID" <<'EOF'
import json, sys
tmp, model = sys.argv[1], sys.argv[2]
json.dump({"model": model, "state": "x", "questions": {f"q{i}": {"type": "noul"} for i in range(65)}}, open(f"{tmp}/many.json", "w"))
json.dump({"model": model, "state": "x", "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}}, open(f"{tmp}/wide.json", "w"))
nested = "[" * 100 + "[" + ",".join(["null"] * 1000) + "]" + "]" * 100
open(f"{tmp}/fanout.json", "w").write('{"model": "%s", "state": %s, "questions": {"q": {"type": "noul"}}}' % (model, nested))
open(f"{tmp}/deep.json", "w").write('{"model": "%s", "state": %s, "questions": {"q": {"type": "noul"}}}' % (model, "[" * 100_000 + "]" * 100_000))
json.dump({"model": model, "state": "x" * (5 << 20), "questions": {"q": {"type": "noul"}}}, open(f"{tmp}/big.json", "w"))
EOF
post_file() { curl -s -H 'Expect:' -o "$TMP/err.json" -w "$2" -m 120 -X POST "localhost:$PORT/v1/decisions" -H 'content-type: application/json' --data-binary @"$1"; }
check "65 questions -> 400" "$(post_file "$TMP/many.json" '%{http_code}')" "400"
check "  ... naming the limit" "$(grep -c 'limit 64' "$TMP/err.json")" "1"
check "256 options -> 400" "$(post_file "$TMP/wide.json" '%{http_code}')" "400"
check "5 KB state that renders to MBs -> 400" "$(post_file "$TMP/fanout.json" '%{http_code}')" "400"
check "  ... naming the render budget" "$(grep -c 'render to more than' "$TMP/err.json")" "1"
check "100k-deep state -> 400" "$(post_file "$TMP/deep.json" '%{http_code}')" "400"
check "5 MB body -> 413" "$(post_file "$TMP/big.json" '%{http_code}')" "413"
check "server still healthy" "$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/health")" "200"

echo "=== questions never see each other ==="
ISO="$(python3 - "$PORT" "$MODEL_ID" <<'EOF'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
def post(qs, state="I was charged twice for my order and want a refund today"):
    body = json.dumps({"model": model, "state": state, "questions": qs}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/decisions", body, {"Content-Type": "application/json"}), timeout=300))["answers"]
qs = {f"q{i}": {"type": "choice", "instructions": "Which team? " + "Read every detail. " * (i * 7 % 40), "criteria": {"billing": None, "sales": None, "tech": None}} for i in range(12)}
together = post(qs)
reversed_ = post(dict(reversed(list(qs.items()))))
alone = {k: post({k: v})[k] for k, v in qs.items()}
_ = post({"x": {"type": "noul"}}, state="a completely different state in between")
again = post(qs)
diff = max(abs(a[k]["probabilities"][l] - b[k]["probabilities"][l]) for a, b in ((together, alone), (together, reversed_), (together, again)) for k in qs for l in ("billing", "sales", "tech"))
print("same" if list(together) == list(qs) and list(reversed_) == list(reversed(list(qs))) and diff == 0 else f"differ {diff}")
EOF
)"
check "12 mixed-length questions == each alone == reversed == after another state (identical HTTP output)" "$ISO" "same"

echo "=== concurrent requests and a failed request in between ==="
CONC="$(python3 - "$PORT" "$MODEL_ID" <<'EOF'
import json, sys, threading, urllib.request, urllib.error
port, model = sys.argv[1], sys.argv[2]
def post(state, qs):
    body = json.dumps({"model": model, "state": state, "questions": qs}).encode()
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/decisions", body, {"Content-Type": "application/json"}), timeout=300))["answers"]
    except urllib.error.HTTPError as e:
        return e.code
qs = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "sales": None, "tech": None}},
      "urgent": {"type": "noul", "instructions": "Is it urgent?"}}
reqs = [(f"ticket {i}: charged {i + 1} times, refund please", qs) for i in range(8)]
reqs.append(("x", {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}))  # 400 alone, the others still answer
serial = [post(*r) for r in reqs]
out = [None] * len(reqs)
def go(i): out[i] = post(*reqs[i])
th = [threading.Thread(target=go, args=(i,)) for i in range(len(reqs))]
[t.start() for t in th]; [t.join() for t in th]
print("same" if out == serial and serial[-1] == 400 and all(isinstance(s, dict) for s in serial[:-1]) else f"differ {out[-1]} {serial[-1]}")
EOF
)"
check "8 concurrent requests + 1 bad one: each answers as it does alone" "$CONC" "same"

echo "=== choice criteria as a list (Laya's shape) ==="
LIST="$(python3 - "$PORT" "$MODEL_ID" <<'EOF'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
def ask(crit):
    body = json.dumps({"model": model, "state": "Charged twice, refund please.",
                       "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": crit}}}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/decisions", body, {"Content-Type": "application/json"}), timeout=120))["answers"]
print("same" if ask(["billing", "sales", "tech"]) == ask({"billing": None, "sales": None, "tech": None}) else "differ")
EOF
)"
check "a list of labels answers exactly like {label: null}" "$LIST" "same"

echo "=== latency (median of 30, 1 question, warm) ==="
LAT="$(python3 - "$PORT" "$MODEL_ID" <<'EOF'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
body = json.dumps({"model": model, "state": {"subject": "Charged twice", "body": "Please refund the duplicate today."},
                   "questions": {"refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}}}).encode()
ts = []
for _ in range(31):
    t = time.time(); urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/decisions", body, {"Content-Type": "application/json"}), timeout=120).read(); ts.append((time.time() - t) * 1000)
print(f"{sorted(ts[1:])[15]:.0f}")
EOF
)"
echo "  median $LAT ms"

check "no MLX error in the server log" "$(grep -c '\[mlx\]' "$LOG")" "0"
check "Kev engine logged ready" "$(grep -c 'Kev engine ready' "$LOG")" "1"
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
