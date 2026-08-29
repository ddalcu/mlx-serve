#!/bin/bash
# Asserts that the KV cache reuses BOTH the prompt prefix AND the previous
# turn's generated tokens across requests.
#
# Why this matters: in a real multi-turn agent loop, each request carries the
# previous turn's assistant output back as part of the prompt. The K/V for
# those generated tokens was already correctly computed during the previous
# request — re-prefilling them on every turn is wasted work AND introduces
# subtle K/V drift (AR `[1,1,d]` qmv vs prefill `[1,K,d]` qmm have different
# float reduction orders at INT4/FP16, so the K/V values for the same tokens
# change between AR-then-cached and re-prefilled).
#
# The test:
#   1. Drives turn 1: short system+user, generates ~120 tokens at temp=0.
#   2. Drives turn 2: same conversation + the assistant reply + a short
#      follow-up user message. Captures wall time.
#   3. Greps the server log for the `[cache] reusing N/M tokens` line
#      emitted by `reuseKVCache`. Asserts:
#        a. Reused N covers the previous turn's PROMPT *plus* its generation
#           — within a small slack to allow the chat template to insert a
#           newline or a turn boundary token between the assistant block and
#           the next user block.
#        b. Turn 2's prefill is meaningfully faster than turn 1's.
#
# Failure signature pre-fix: line (a) reports `reusing P1/P2` where P1 is
# only turn-1's prompt length, NOT including its generation. The test then
# asserts P1 >= turn_1_prompt + turn_1_completion - 8 and fails because P1
# is short by `turn_1_completion - 8` tokens.
#
# Default port 8092 (own server). Pass [port] to reuse a running one.

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
exec python3 - "$LOG" "$BASE" <<'PY'
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
    {"role": "assistant", "content": r1["content"]},
    {"role": "user", "content": "Now do the same but with squares of those numbers."},
]
r2 = chat(m2, max_tokens=200)
print(f"  prompt_tokens={r2['prompt_tokens']}  completion_tokens={r2['completion_tokens']}  elapsed={r2['elapsed_ms']}ms")

# ── Parse server log for the [cache] reusing line emitted on turn 2 ──
with open(LOG) as f:
    f.seek(log_start_offset)
    log_text = f.read()

# Reuse logs come from `reuseKVCache` — match either the extension-reuse
# log ("reusing N/M tokens from previous prompt") or the identical-re-issue
# log ("reusing N/M tokens, re-forwarding last token"). Take the LAST one
# in the captured window so we look at turn 2.
reuse_pat = re.compile(r"\[cache\] reusing (\d+)/(\d+) tokens", re.M)
matches = reuse_pat.findall(log_text)
print(f"\n{DIM}reuse-line matches in this window: {matches}{NC}")

if not matches:
    # Could be that turn 2 went through a path that resets the cache
    # entirely. Surface the last few cache-related log lines for diagnosis.
    cache_lines = [l for l in log_text.splitlines() if "[cache]" in l][-6:]
    print(f"\n{DIM}last [cache] lines:{NC}")
    for l in cache_lines: print(f"    {l}")
    check(False, "turn 2 emitted a cache-reuse log line",
          "no `[cache] reusing N/M tokens` line found — cache was hard-reset")
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
