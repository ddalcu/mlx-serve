#!/bin/bash
# test_cancel_commit.sh — cancelled requests preserve their committed KV.
#
# Two contracts, one per cancellation phase:
#   [1] decode-phase cancel: a streaming request killed mid-generation
#       (client disconnect) must COMMIT prompt + emitted tokens to the hot
#       prefix cache — `complete()` pulls the slot straight into the cleanup
#       queue, so without the drain-hook commit its KV dies with the slot.
#       Proven by an identical re-issue reporting `cached_tokens` ≈ the full
#       prompt (full-match reuse) where it previously reported ~0.
#   [2] prefill-phase cancel (non-hybrid models only): a long prompt killed
#       while still prefilling must commit the FORWARDED PREFIX (chunk-
#       aligned). Hybrids are excluded by design — their stride SSM
#       checkpoints die with the failed Generator init, and a checkpoint-less
#       hybrid entry restores as a cold miss while occupying an LRU slot —
#       so the section self-detects hybrid configs (GDN / layer_types) and
#       SKIPs with an explanation.
#       The killed stream's captured body tells us which phase we were in:
#       no `data:` content chunks ⇒ still prefilling (the strict
#       "committed N/M prompt tokens from a cancelled prefill" log line is
#       then REQUIRED); content chunks ⇒ prefill had finished and the
#       decode-cancel contract of [1] covers it (section skipped, not failed).
#
# Usage: ./tests/test_cancel_commit.sh [model_dir] [port]
#   Starts its own server. Default model: Gemma 4 E4B 4-bit.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-11272}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_cancel_commit.log
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS+1)); echo -e "${GREEN}PASS${NC}: $desc"
    else
        FAIL=$((FAIL+1)); echo -e "${RED}FAIL${NC}: $desc"
    fi
}

[[ -x "$BINARY" ]] || { echo "Build first (zig build -Doptimize=ReleaseFast)" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "Model not found: $MODEL (SKIP)" >&2; exit 0; }

# Long system prompt (well past the ~40-token template prologue and the
# 256-token cancelled-prefill floor) so a cache hit is unambiguous.
SYSTEM_PROMPT="You are an expert software engineer assistant. You provide concise, technically correct answers. You explain trade-offs when relevant. You always cite specific function names, file paths, or line numbers when discussing code. You prefer concrete examples over abstract advice. You do not pad your answers with hedges or apologies. You assume the user is also a software engineer. You write in Markdown when formatting helps. You keep code blocks small and self-contained. You ask clarifying questions only when truly necessary. You favor depth over breadth in your explanations. You show your reasoning when it would help the reader. You never invent APIs. You state uncertainty explicitly when you are unsure."

# A much longer one (~5-6k tokens) so prefill takes long enough for a
# mid-prefill kill to land inside it even on fast small models.
LONG_SYSTEM=""
for i in $(seq 1 60); do
    LONG_SYSTEM="$LONG_SYSTEM Chapter $i. $SYSTEM_PROMPT"
done

boot_server() {
    rm -f "$LOG"
    "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --ctx-size 16384 --prefill-chunk 1024 --kv-quant 4 --prefix-cache-entries 4 \
        > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERR: server died" >&2; return 1; }
        sleep 1
    done
    echo "ERR: server never became healthy" >&2; return 1
}

# Model load is lazy on a headless boot: warm it with a tiny UNRELATED
# request (different prompt ⇒ different cache key ⇒ no interference) before
# timing anything. Generous timeout — the first request pays the full
# multi-GB checkpoint load.
warmup_model() {
    curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0.0}' \
        >/dev/null 2>&1
}

stop_server() {
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}

trap 'stop_server; true' EXIT

body_chat() {
    local sys="$1" stream="$2" maxtok="$3"
    jq -nc --arg s "$sys" \
        '{model:"x",messages:[{role:"system",content:$s},{role:"user",content:"Tell me about cache invalidation."}],max_tokens:$mt,temperature:0.0,stream:$st,enable_thinking:false}' \
        --argjson mt "$maxtok" --argjson st "$stream"
}

# ─────────────────────────────────────────────────────────────────────────
# [1] decode-phase cancel
# ─────────────────────────────────────────────────────────────────────────
decode_cancel_section() {
    echo "== [1] decode-phase cancel =="
    boot_server || return 1
    warmup_model

    # Fire a streaming generation and kill the client mid-decode. The short
    # prompt prefills in well under a second on any model, and a thinking-off
    # answer can hit EOS in a few seconds on a small one, so the kill lands
    # early.
    local body
    body="$(body_chat "$SYSTEM_PROMPT" true 2000)"
    curl -sN --max-time 3 -o /tmp/test_cancel_partial.sse \
        -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" 2>/dev/null
    local curl_rc=$?
    # 28 = timeout kill (what we want); 18/56/55 also fine (partial/close).
    if [ "$curl_rc" -eq 0 ]; then
        check "killed stream actually got cut early (rc=$curl_rc, needed >0)" 0
        stop_server; return 1
    fi
    # Some decode tokens must have flown before the kill, else this is not a
    # decode-phase cancel. grep for content chunks (choices with delta).
    if ! grep -qE '"(content|reasoning_content)":"[^"]' /tmp/test_cancel_partial.sse 2>/dev/null; then
        check "decode tokens observed before the kill (mid-decode cancel)" 0
        stop_server; return 1
    fi

    # Give the server a moment to notice the dead socket, cancel the slot
    # and drain the cleanup queue (the commit happens there).
    sleep 4

    # Identical re-issue: full prompt should hit the committed entry.
    local resp cached
    resp="$(curl -sf --max-time 120 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(body_chat "$SYSTEM_PROMPT" false 1)")"
    cached="$(echo "$resp" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0' 2>/dev/null)"
    check "identical re-issue after decode-cancel reuses cached KV (cached_tokens=$cached)" \
        "$([[ "${cached:-0}" -ge 100 ]] && echo 1 || echo 0)"

    grep -q "\[hot-cache\] reused" "$LOG" && ok=1 || ok=0
    check "server log shows [hot-cache] reused" "$ok"

    stop_server
}

# ─────────────────────────────────────────────────────────────────────────
# [2] prefill-phase cancel (non-hybrid only)
# ─────────────────────────────────────────────────────────────────────────
prefill_cancel_section() {
    echo "== [2] prefill-phase cancel =="

    # Hybrid detection: GDN / linear layer_types / mamba ⇒ the cancelled
    # prefill commit is intentionally OFF (checkpoints die with the failed
    # Generator init; a checkpoint-less hybrid entry restores as a cold miss
    # while polluting the LRU).
    local cfg="$MODEL/config.json"
    if [ -f "$cfg" ] && grep -qE 'gated_deltanet|"linear"|mamba|full_attention_interval' "$cfg"; then
        echo "SKIP: hybrid architecture detected — cancelled-prefill commit is excluded by design"
        return 0
    fi

    boot_server || return 1
    warmup_model

    # ~8k-token prompt at --prefill-chunk 1024 (the boot flag): several
    # chunks are still ahead at kill time on any model, and the chunk loop
    # only polls the cancel flag at chunk boundaries — one 8k chunk would
    # complete the whole prefill first. On very fast models the kill may
    # still land in decode, which the captured stream tells us (see below);
    # that case is section [1]'s contract.
    local body
    body="$(body_chat "$LONG_SYSTEM" true 2000)"
    curl -sN --max-time 2 -o /tmp/test_cancel_prefill.sse \
        -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" 2>/dev/null
    local curl_rc=$?

    sleep 8  # keepalive-detection window + cleanup drain

    local cached
    cached="$(curl -sf --max-time 180 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(body_chat "$LONG_SYSTEM" false 1)" \
        | jq -r '.usage.prompt_tokens_details.cached_tokens // 0' 2>/dev/null)"

    # A prefill that finishes inside the 5 s keepalive silence window is
    # only seen as disconnected at the first token write: no content in the
    # stream, yet the cache holds a FULL entry (e4b prefills 8k in ~3 s).
    if grep -qE '"(content|reasoning_content)":"[^"]' /tmp/test_cancel_prefill.sse 2>/dev/null \
       || grep -q "\[hot-cache\] full reuse" "$LOG"; then
        # Prefill completed before the kill landed — the cancel hit decode
        # phase; not this section's contract (and [1] covers it).
        echo "NOTE: kill landed in decode phase (prompt prefilled too fast) — cached_tokens=$cached reported for information"
        check "cancelled request KV still preserved (cached_tokens=$cached)" \
            "$([[ "${cached:-0}" -ge 100 ]] && echo 1 || echo 0)"
    else
        # Genuinely mid-prefill: the forwarded-prefix commit is REQUIRED.
        check "mid-prefill kill reuses forwarded prefix (cached_tokens=$cached)" \
            "$([[ "${cached:-0}" -ge 256 ]] && echo 1 || echo 0)"
        grep -q "from a cancelled prefill" "$LOG" && ok=1 || ok=0
        check "server log shows the cancelled-prefill commit" "$ok"
    fi

    stop_server
}

decode_cancel_section
prefill_cancel_section

echo
echo "RESULT: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]] || exit 1
exit 0
