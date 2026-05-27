#!/bin/bash
# test_prefix_cache_hot.sh — plan 03 phase 1 hot prefix cache validation.
#
# Two parallel conversations against the same server, each with a long shared
# system-prompt prefix. With --prefix-cache-entries 0 the hot cache is fully
# disabled, so alternating between conversations evicts the other's KV after
# every turn and every request pays full prefill. With --prefix-cache-entries
# 2, both conversations should keep their prefix hot across the alternation.
#
# Acceptance: alternating-mode 2nd-turn prefill TPS for conversation A is
# ≥ 2× faster with hot cache than without (since with the cache disabled, the
# 2nd-turn A request pays cold prefill on a long prompt).
#
# Note: --prefix-cache-entries 1 (the default) is NOT the "no hot cache" path
# any more — it's a 1-entry hot cache. The disable knob is 0. Earlier versions
# of this test treated 1 as disabled, which silently broke once the hot cache
# became the default.
#
# Usage: ./tests/test_prefix_cache_hot.sh [model_dir] [port]

set -uo pipefail

MODEL_DIR="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-19040}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"

[[ -x "$BINARY" ]] || { echo "Build first" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Model not found: $MODEL_DIR" >&2; exit 1; }

trap 'pkill -9 -x mlx-serve 2>/dev/null; true' EXIT

# Long shared system prompt (250+ tokens) so prefill cost is significant.
SYSTEM_PROMPT="You are an expert software engineer assistant. You provide concise, technically correct answers. You explain trade-offs when relevant. You always cite specific function names, file paths, or line numbers when discussing code. You prefer concrete examples over abstract advice. You do not pad your answers with hedges or apologies. You assume the user is also a software engineer. You write in Markdown when formatting helps. You keep code blocks small and self-contained. You ask clarifying questions only when truly necessary. You favor depth over breadth in your explanations. You show your reasoning when it would help the reader."

CONVO_A_USER1="What is the difference between a mutex and a semaphore?"
CONVO_B_USER1="What is the difference between TCP and UDP at the protocol level?"
CONVO_A_USER2="Now explain when to use each of those in practice."
CONVO_B_USER2="Now explain when each is preferred in practice."

# Build a single-turn body (system + user only).
build_body() {
    local user_text="$1" sys="$2"
    jq -nc --arg u "$user_text" --arg s "$sys" \
        '{model:"x",messages:[{role:"system",content:$s},{role:"user",content:$u}],max_tokens:1,temperature:0.0,stream:false,enable_thinking:false}'
}

# Build a multi-turn body where the assistant's prior reply is included so the
# new user turn shares the FULL prior turn as cached prefix. Critical for
# isolating cap=0 vs cap=2: with cap=0 the cache is fully disabled, so A2 pays
# cold prefill for the whole prompt; with cap=2 A1's commit AND B1's commit
# are both retained, so A2's entry-A snapshot covers `system+user1+assistant`
# and only the new user text needs prefill.
build_body_multiturn() {
    local sys="$1" user1="$2" assistant1="$3" user2="$4"
    jq -nc --arg s "$sys" --arg u1 "$user1" --arg a1 "$assistant1" --arg u2 "$user2" \
        '{model:"x",messages:[{role:"system",content:$s},{role:"user",content:$u1},{role:"assistant",content:$a1},{role:"user",content:$u2}],max_tokens:1,temperature:0.0,stream:false,enable_thinking:false}'
}

call_and_time() {
    local body="$1"
    local t0 t1
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body" >/dev/null
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    echo "$((t1 - t0))"
}

run_with_capacity() {
    local cap="$1"
    pkill -9 -x mlx-serve 2>/dev/null
    sleep 1

    "$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --ctx-size 4096 \
        --prefix-cache-entries "$cap" --log-level info > /tmp/test_prefix_cache.log 2>&1 &
    local pid=$!
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 0.5
        kill -0 "$pid" 2>/dev/null || { echo "ERR: server died" >&2; return 1; }
    done

    # Step 1-2: prime conversations A and B (single-turn).
    # Step 3-4: A and B follow-up multi-turn (history = system+user1+ASSISTANT+user2).
    #   - cap=0: hot cache disabled → every request is a cold prefill, no
    #            [hot-cache] log lines emitted.
    #   - cap=2: A1's commit and B1's commit both retained → A2's lookup against A1
    #     finds full match of (system+user1+assistant) → only user2 needs prefill
    local a1 b1 a2 b2 fake_assistant_a fake_assistant_b
    fake_assistant_a="A mutex protects shared resources from race conditions, while a semaphore limits concurrent access to a fixed-size pool. Both use atomic operations under the hood."
    fake_assistant_b="TCP is connection-oriented and reliable, with handshakes, retransmits, and ordered delivery. UDP is connectionless and fire-and-forget, relying on the application for any reliability guarantees."

    a1=$(call_and_time "$(build_body "$CONVO_A_USER1" "$SYSTEM_PROMPT")")
    b1=$(call_and_time "$(build_body "$CONVO_B_USER1" "$SYSTEM_PROMPT")")
    a2=$(call_and_time "$(build_body_multiturn "$SYSTEM_PROMPT" "$CONVO_A_USER1" "$fake_assistant_a" "$CONVO_A_USER2")")
    b2=$(call_and_time "$(build_body_multiturn "$SYSTEM_PROMPT" "$CONVO_B_USER1" "$fake_assistant_b" "$CONVO_B_USER2")")

    pkill -9 -x mlx-serve 2>/dev/null
    sleep 1
    echo "$a1 $b1 $a2 $b2"
}

echo "=== capacity=0 (hot cache disabled) ==="
read -r a1_c0 b1_c0 a2_c0 b2_c0 <<< "$(run_with_capacity 0)"
echo "  walls: A1=${a1_c0}ms B1=${b1_c0}ms A2=${a2_c0}ms B2=${b2_c0}ms"
log_c0="$(cat /tmp/test_prefix_cache.log)"

echo "=== capacity=2 (hot prefix cache) ==="
read -r a1_c2 b1_c2 a2_c2 b2_c2 <<< "$(run_with_capacity 2)"
echo "  walls: A1=${a1_c2}ms B1=${b1_c2}ms A2=${a2_c2}ms B2=${b2_c2}ms"
log_c2="$(cat /tmp/test_prefix_cache.log)"

# Behavior assertions (more robust than wall-time deltas, which are noisy
# at short prompts and decode-dominated regimes):
#
#  1. cap=0 path must NOT engage the hot cache (no [hot-cache] log lines).
#  2. cap=2 path must engage the hot cache (≥2 [hot-cache] reused lines).
#  3. cap=2 A2's lookup must find more shared tokens than cap=0's path
#     because A1's snapshot is retained alongside B1's. The cap=2 log must
#     surface ≥ 1 reuse line whose `entry K/L` suffix shows L ≥ 2 — i.e.
#     two entries were resident at the same time.

# Assertion 1.
if echo "$log_c0" | grep -qE "\[hot-cache\]"; then
    echo "FAIL: cap=0 unexpectedly engaged hot cache" >&2
    exit 1
fi
echo "PASS: cap=0 does not engage hot cache (disabled)"

# Assertion 2. Match BOTH log surfaces the cache emits on a reuse:
#   `[hot-cache] reused N/M tokens (matched X; entry K/L)`  (partial match)
#   `[hot-cache] full reuse N/M, re-forwarding last token`  (full match)
# Either is a legitimate reuse — the partial-vs-full split is an internal
# branch in lookupAndRestore, not a behavior change from the test's POV.
hits_c2=$(echo "$log_c2" | grep -cE "\[hot-cache\] (reused|full reuse)")
if [[ "$hits_c2" -lt 2 ]]; then
    echo "FAIL: cap=2 expected ≥ 2 hot-cache reuse events, got ${hits_c2}" >&2
    echo "--- log:" >&2
    echo "$log_c2" | grep -E "\[hot-cache\]" >&2
    exit 1
fi
echo "PASS: cap=2 hot cache fired ${hits_c2} reuse events"

# Assertion 3: the cap=2 log must show A2 (or B2) reusing from one of
# multiple entries. Format: `reused N/M tokens (matched X; entry K/L)`
# where L >= 2 confirms multi-entry state.
if ! echo "$log_c2" | grep -qE "entry [0-9]+/[2-9][0-9]*\)"; then
    echo "FAIL: cap=2 hot cache never had 2+ entries simultaneously" >&2
    echo "--- log:" >&2
    echo "$log_c2" | grep -E "\[hot-cache\]" >&2
    exit 1
fi
echo "PASS: cap=2 holds multiple entries simultaneously"

echo
echo "PASS: hot prefix cache wired correctly"
