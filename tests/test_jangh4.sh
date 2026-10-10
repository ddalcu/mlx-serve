#!/usr/bin/env bash
# JANGH4 (qwen4_exp with jangtq2 codebook experts) live on the bundle as published.
# Boots it with --no-mtp --no-pld (greedy is then byte-stable), then: the load
# and its n-gram table, a short greedy answer on the decode kernels, a prompt
# past 96 tokens on the expert-sorted GEMM (both asserted through the
# `[jangtq2] ... engaged` lines), and that prompt again: restored from the
# prefix cache, same bytes. [5] reboots with MTP on (the head's affine experts
# come from the bundle's fused tensor). SKIPs without the bundle.
#   JANGH4_MODEL=<bundle dir> ./tests/test_jangh4.sh [port]
set -u
MODEL="${JANGH4_MODEL:-$HOME/.mlx-serve/models/JANGQ-AI/Qwen3.8-Flash-Next-JANGH4}"
PORT="${1:-11412}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
LOG="$HOME/claude-tmp/jangh4-live/server-$PORT.log"
mkdir -p "$(dirname "$LOG")"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no JANGH4 bundle at $MODEL"; exit 0; }
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
U="http://127.0.0.1:$PORT"
boot() {
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info --ctx-size 32768 --no-pld "$@" > "$LOG" 2>&1 &
  SPID=$!
  for _ in $(seq 1 600); do curl -s "$U/health" >/dev/null 2>&1 && grep -q "Model ready" "$LOG" && return 0; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }; sleep 2; done
  echo "server not ready"; exit 1
}
chat() { curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$1"; }
content() { python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"; }
boot --no-mtp
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
echo "[1] the bundle loads as qwen4_exp with its in-shard n-gram table"
arch=$(curl -s "$U/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['meta'].get('architecture',''))")
check "architecture" "$arch" "qwen4_exp"
check "n-gram table log" "$(grep -c '\[qwen4\] n-gram table' "$LOG")" "1"
echo "[2] greedy short answer on the decode kernels"
ans=$(chat '{"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":64,"temperature":0,"enable_thinking":false}' | content)
echo "  -> $(echo "$ans" | tr '\n' ' ' | cut -c1-60)"
check "mentions Paris" "$(echo "$ans" | grep -ci paris | sed 's/^[1-9][0-9]*$/1/')" "1"
check "decode engaged line" "$(grep -c '\[jangtq2\] decode engaged' "$LOG")" "1"
echo "[3] a prompt past 96 tokens on the expert-sorted GEMM"
long=$(python3 -c "
import json
filler=' '.join(f'Fact {i}: the river near town number {i} runs east.' for i in range(150))
print(json.dumps({'messages':[{'role':'user','content':filler+' The secret code is HERON-77. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':48,'temperature':0,'enable_thinking':False}))")
r1=$(chat "$long")
a1=$(echo "$r1" | content)
p1=$(echo "$r1" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])")
echo "  prompt_tokens $p1 -> $(echo "$a1" | tr '\n' ' ' | cut -c1-60)"
check "prompt past the gather width" "$(python3 -c "print(1 if $p1 > 96 else 0)")" "1"
check "prefill engaged line" "$(grep -c '\[jangtq2\] prefill engaged' "$LOG")" "1"
check "needle recovered" "$(echo "$a1" | grep -c 'HERON-77')" "1"
echo "[4] the same prompt again: restored from the prefix cache, same bytes"
r2=$(chat "$long")
cached=$(echo "$r2" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens_details']['cached_tokens'])")
echo "  cached_tokens $cached of $p1"
check "prefix restored" "$(python3 -c "print(1 if $cached > 0 else 0)")" "1"
check "greedy answer byte-identical" "$( [ "$(echo "$r2" | content)" = "$a1" ] && echo 1 || echo 0)" "1"
echo "[5] MTP on: the head drafts and the answer holds"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
boot
ans=$(chat '{"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":64,"temperature":0,"enable_thinking":false,"enable_mtp":true}' | content)
echo "  -> $(echo "$ans" | tr '\n' ' ' | cut -c1-60)"
check "mentions Paris" "$(echo "$ans" | grep -ci paris | sed 's/^[1-9][0-9]*$/1/')" "1"
check "mtp engaged" "$(grep -c 'spec-stats\] mode=mtp' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
echo "passed $pass failed $fail"
[ "$fail" = 0 ]
