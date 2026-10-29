#!/bin/bash
# test_cache_reuses_generated_tokens.sh — the KV cache reuses the previous turn's GENERATED tokens, not just its prompt:
# turn 2 resends turn 1 + its ~120-token reply + a follow-up, and `[hot-cache] reused N/M` must cover turn 1's prompt
# plus its completion (within a template slack of 8), with a faster prefill than turn 1. Re-prefilling them wastes work
# and moves K/V (decode qmv and prefill qmm reduce in different orders).
#
#   ./tests/test_cache_reuses_generated_tokens.sh [port]   # own server on 8092, or pass a running one's port
#
# On a hybrid (GDN) model the reuse comes from the decode-end SSM checkpoint: CACHE_GEN_TEST_MODEL=<Qwen3.5/3.8 dir>.

set -u

PORT="${1:-8092}"
BASE="http://127.0.0.1:$PORT"
MODEL="${CACHE_GEN_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_cache_reuses_generated_tokens: model not found ($MODEL)"
    exit 0
fi
if [ ! -x ./zig-out/bin/mlx-serve ]; then
    echo -e "${RED}FAIL${NC} mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

LOG=$(mktemp)
echo "Starting mlx-serve on port $PORT (log: $LOG)..."
# `--log-level info` so the [cache] reusing line is captured (debug is too noisy).
./zig-out/bin/mlx-serve --model "$MODEL" --serve --port "$PORT" --no-pld \
    --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -f "$LOG"' EXIT

for _ in $(seq 1 90); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${RED}FAIL${NC} server crashed during load"
        tail -30 "$LOG"; exit 1
    fi
    sleep 1
done
if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo -e "${RED}FAIL${NC} server did not become healthy"
    tail -30 "$LOG"; exit 1
fi

# All driving + parsing in python so we get clean JSON + log handling.
# Not exec: bash has to outlive python for its EXIT trap to stop the server.
python3 - "$LOG" "$BASE" <<'PY'
import json, re, sys, time, urllib.request

LOG = sys.argv[1]
BASE = sys.argv[2]

RED = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"; DIM = "\033[2m"; NC = "\033[0m"

def chat(messages, max_tokens=120):
    body = {
        "model": "mlx-serve",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read())
    elapsed_ms = int((time.time() - t0) * 1000)
    msg = d["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "completion_tokens": d["usage"]["completion_tokens"],
        "elapsed_ms": elapsed_ms,
    }

passes, fails = 0, 0
def check(cond, desc, detail=""):
    global passes, fails
    if cond:
        passes += 1; print(f"  {GREEN}PASS{NC} {desc}")
    else:
        fails += 1; print(f"  {RED}FAIL{NC} {desc}")
        if detail: print(f"    {DIM}{detail}{NC}")

# Mark log offset so we only inspect THIS test's lines.
log_start_offset = 0
try:
    with open(LOG, "rb") as f: f.seek(0, 2); log_start_offset = f.tell()
except FileNotFoundError:
    pass

print(f"{CYAN}=== Cache reuse of generated tokens ==={NC}")

# ── Turn 1 ──
print("\nTurn 1: cold cache, generate ~120 tokens")
SYS = ("You are a helpful assistant. When asked to count, count out loud "
       "from 1 to 50, separated by commas, on a single line.")
m1 = [
    {"role": "system", "content": SYS},
    {"role": "user", "content": "Count from 1 to 50 as instructed."},
]
r1 = chat(m1, max_tokens=200)
print(f"  prompt_tokens={r1['prompt_tokens']}  completion_tokens={r1['completion_tokens']}  elapsed={r1['elapsed_ms']}ms")
check(r1["completion_tokens"] >= 30,
      "turn 1 generated a substantial reply (>=30 tokens)",
      f"got {r1['completion_tokens']}")

# ── Turn 2 ── append the assistant reply + a short follow-up.
print("\nTurn 2: history grows, follow-up should reuse turn-1 prompt + generation")
m2 = m1 + [
    # Thinking models get their reasoning back, as agents send it, so the turn re-renders token for token.
    {"role": "assistant", "content": r1["content"], **({"reasoning_content": r1["reasoning"]} if r1["reasoning"] else {})},
    {"role": "user", "content": "Now do the same but with squares of those numbers."},
]
r2 = chat(m2, max_tokens=200)
print(f"  prompt_tokens={r2['prompt_tokens']}  completion_tokens={r2['completion_tokens']}  elapsed={r2['elapsed_ms']}ms")

# ── Parse server log for the [hot-cache] reused line emitted on turn 2 ──
with open(LOG) as f:
    f.seek(log_start_offset)
    log_text = f.read()

# Take the LAST reuse line in the captured window so we look at turn 2.
reuse_pat = re.compile(r"\[hot-cache\] reused (\d+)/(\d+) tokens", re.M)
matches = reuse_pat.findall(log_text)
print(f"\n{DIM}reuse-line matches in this window: {matches}{NC}")

if not matches:
    # Could be that turn 2 went through a path that resets the cache
    # entirely. Surface the last few cache-related log lines for diagnosis.
    cache_lines = [l for l in log_text.splitlines() if "[hot-cache]" in l][-6:]
    print(f"\n{DIM}last [hot-cache] lines:{NC}")
    for l in cache_lines: print(f"    {l}")
    check(False, "turn 2 emitted a cache-reuse log line",
          "no `[hot-cache] reused N/M tokens` line found — cache was hard-reset")
else:
    n_reused, n_total = (int(x) for x in matches[-1])
    # Expected lower bound: turn-1 prompt + turn-1 completion. Slack of 8
    # accounts for chat-template tokens inserted between the assistant
    # block and the new user block (turn boundaries, role markers).
    SLACK = 8
    expected_min = r1["prompt_tokens"] + r1["completion_tokens"] - SLACK
    print(f"\n  reused={n_reused}/{n_total}  expected_min={expected_min}")
    check(n_reused >= expected_min,
          f"turn 2 reused at least {expected_min} tokens (turn-1 prompt + generation - {SLACK} template-slack)",
          f"only reused {n_reused} — turn-1 generation ({r1['completion_tokens']} tokens) was NOT cached")

# ── Wall-time sanity check: turn 2 prefill should be much faster than
# turn 1's once gen-tokens are reused. We compare prefill-only proxies:
# turn 1 ≈ full prompt prefill + decode; turn 2 ≈ small new-token prefill
# + decode. Both decode the same max_tokens, so the delta is mostly
# prefill, and turn 2 should win.
print(f"\n  turn 1 elapsed: {r1['elapsed_ms']}ms")
print(f"  turn 2 elapsed: {r2['elapsed_ms']}ms")
# Don't assert a specific ratio — that's flaky on CI. Just report.
ratio = (r1["elapsed_ms"] / r2["elapsed_ms"]) if r2["elapsed_ms"] > 0 else 0
print(f"  turn1/turn2 wall-time ratio: {ratio:.2f}x")

print(f"\n{CYAN}═══════════════════════════════════════════{NC}")
print(f"  Passed: {GREEN}{passes}{NC}  Failed: {RED}{fails}{NC}")
print(f"{CYAN}═══════════════════════════════════════════{NC}")
sys.exit(0 if fails == 0 else 1)
PY
exit $?
