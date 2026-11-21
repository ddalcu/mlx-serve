#!/bin/bash
# Inkling 2 Mini (inkling2_loop, the inkling2 plugin) live end-to-end on a local pack:
#
#   ./tests/test_inkling2.sh /path/to/Inkling-2-Mini-4bit
#   INKLING2_TEST_MODEL=/path/to/pack ./tests/test_inkling2.sh   (default: mlx-community/Inkling-2-Mini-4bit if found)
#
#   [1] /v1/models lists the pack                [4] an image part is a 400 naming images
#   [2] greedy raw completion: " Berlin"         [5] reasoning_effort: 200 with non-empty content
#   [3] prefix cache: the same chat twice reuses
#       its prefix and returns the same tokens;
#       a chat sharing a long prefix restores at a checkpoint and equals a cold server's answer
#
# --ctx-size 32768: at the default context (85% of the memory ceiling) the KV bill leaves the hot cache 0 MB and [3]
# would test nothing. The cold reference for [3] is a second boot with --prefix-cache-entries 0.
# Hermetic counterparts: src/plugins/inkling2/proof_test.zig + served_test.zig (fixtures and real packs vs mlx-lm).

set -euo pipefail
cd "$(dirname "$0")/.."
source tests/_lib_models.sh
source tests/lib/server.sh

MODEL="${1:-${INKLING2_TEST_MODEL:-$(find_model mlx-community/Inkling-2-Mini-4bit)}}"
if [ -z "$MODEL" ]; then echo "SKIP: no Inkling 2 Mini pack (pass a path or set INKLING2_TEST_MODEL)"; exit 0; fi
[ -f "$MODEL/config.json" ] || { echo "FAIL: $MODEL/config.json not found"; exit 1; }

PORT="${INKLING2_TEST_PORT:-11299}"
BASE="http://127.0.0.1:$PORT"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
SCRATCH=$(mktemp -d "$HOME/claude-tmp/inkling2_test.XXXXXX")
SERVER_PID=""
cleanup() { srv_stop "$SERVER_PID"; }
trap cleanup EXIT

boot() { # log, extra flags...
    local log="$1"; shift
    srv_preclean "$PORT"
    HOME="$SCRATCH" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 \
        --log-level debug "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 300 1 "$SERVER_PID"
}

if ! boot "$SCRATCH/server.log"; then
    echo "FAIL: [1] the server did not come up on $MODEL"; tail -20 "$SCRATCH/server.log"; exit 1
fi

python3 -I - "$BASE" "$SCRATCH" "$MODEL" warm <<'PY' || { echo "logs: $SCRATCH"; exit 1; }
import json, os, re, sys, urllib.request, urllib.error
base, scratch, model, phase = sys.argv[1:5]
log = os.path.join(scratch, "server.log")
fails = []
def ok(name, cond, got=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n  got: {str(got)[:400]}"))
    if not cond: fails.append(name)
def post(path, body):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=900); return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
def chat(content, **kw):
    return post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "temperature": 0, **kw})
def tokens(r):  # the whole greedy answer: thinking and content
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
def reused():
    return re.findall(r"\[hot-cache\] reused (\d+)/(\d+) tokens", open(log, errors="replace").read())

ids = [m["id"] for m in json.load(urllib.request.urlopen(base + "/v1/models"))["data"]]
ok("[1] /v1/models lists the pack", os.path.basename(model.rstrip("/")) in ids, ids)

st, r = post("/v1/completions", {"prompt": "The capital of France is Paris. The capital of Germany is",
                                 "max_tokens": 8, "temperature": 0})
ok("[2] greedy completion starts with ' Berlin'", st == 200 and r["choices"][0]["text"].startswith(" Berlin"), r)

q = "Name three primary colors. Answer briefly."
st1, r1 = chat(q, max_tokens=160)
n = len(reused())
st2, r2 = chat(q, max_tokens=160)
hits = reused()[n:]
ok("[3a] the same chat again reuses its prefix", st2 == 200 and hits and int(hits[-1][0]) > 0
   and r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, (hits, r2.get("usage")))
ok("[3a] and returns the same tokens", st1 == st2 == 200 and tokens(r1) == tokens(r2), (tokens(r1), tokens(r2)))

# A long shared prefix, then a different question: the second chat restores the first one's state mid-prompt.
P = "Here is a log of a ship's voyage. " + " ".join(
    f"Day {i}: the wind came from the {['north', 'south', 'east', 'west'][i % 4]}, the crew counted {i * 7 % 31} "
    f"gulls and mended {i % 5} sails." for i in range(1, 40))
Q2 = P + "\n\nHow many sails were mended in total? Answer briefly."
chat(P + "\n\nOn which day did the crew count the most gulls? Answer briefly.", max_tokens=160)
st, r = chat(Q2, max_tokens=160)
ok("[3b] a shared long prefix is restored", st == 200 and r["usage"]["prompt_tokens_details"]["cached_tokens"] > 512,
   r.get("usage"))
json.dump({"q": Q2, "tokens": tokens(r) if st == 200 else None}, open(os.path.join(scratch, "warm.json"), "w"))

img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
st, r = chat([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
              {"type": "text", "text": "What is in this picture?"}], max_tokens=8)
ok("[4] an image part is a 400 naming images", st == 400 and "image" in r.get("error", {}).get("message", ""), r)

st, r = chat("What is 3*7? Answer with just the number.", max_tokens=400, reasoning_effort="low")
ok("[5] reasoning_effort: 200 with content", st == 200 and (r["choices"][0]["message"].get("content") or "").strip(), r)

sys.exit(1 if fails else 0)
PY

srv_stop "$SERVER_PID"
if ! boot "$SCRATCH/cold.log" --prefix-cache-entries 0; then
    echo "FAIL: [3b] the cold server did not come up"; tail -20 "$SCRATCH/cold.log"; exit 1
fi
python3 -I - "$BASE" "$SCRATCH" <<'PY' || { echo "logs: $SCRATCH"; exit 1; }
import json, os, sys, urllib.request
base, scratch = sys.argv[1:3]
warm = json.load(open(os.path.join(scratch, "warm.json")))
body = {"messages": [{"role": "user", "content": warm["q"]}], "max_tokens": 160, "temperature": 0}
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                                            {"content-type": "application/json"}), timeout=900))
m = r["choices"][0]["message"]
cold = (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
same = r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and cold == warm["tokens"]
print(("PASS" if same else "FAIL") + " [3b] the restored answer equals a cold server's"
      + ("" if same else f"\n  warm: {warm['tokens'][:300]!r}\n  cold: {cold[:300]!r}"))
sys.exit(0 if same else 1)
PY
rm -rf "$SCRATCH"
echo "inkling2: all checks passed"
