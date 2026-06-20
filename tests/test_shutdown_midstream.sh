#!/bin/bash
# Regression: SIGTERM (graceful shutdown) while a streaming request is in flight
# must NOT crash the server. Pre-fix, the accept loop exited and scheduler.deinit
# tore down the slot queues while a detached connection thread was still inside
# Scheduler.complete() → use-after-free SIGSEGV (crash report
# mlx-serve-2026-06-20-141700.ips). Fix: serve() cancels in-flight slots and
# drains connection threads before returning (so deinit runs after they finish).
#
# Each iteration boots a fresh server, fires a long streaming generation, sends
# SIGTERM mid-stream, and asserts the process exits cleanly (not signal 139).
#
# Usage: MODEL=<dir> ./tests/test_shutdown_midstream.sh [port] [iterations]
set -u
MODEL="${MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
PORT="${1:-11455}"
ITERS="${2:-8}"
BIN="./zig-out/bin/mlx-serve"
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; NC=$'\033[0m'

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 0; }
[ -f "$MODEL/config.json" ] || { echo "SKIP: model not found: $MODEL"; exit 0; }

PASS=0; FAIL=0
for i in $(seq 1 "$ITERS"); do
  "$BIN" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 --log-level warn >/tmp/shutdown_srv_$i.log 2>&1 &
  SVPID=$!
  # wait healthy
  up=0
  for _ in $(seq 1 60); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SVPID" 2>/dev/null || break
    sleep 0.5
  done
  if [ "$up" != 1 ]; then echo "  iter $i: server didn't start, skipping"; kill -9 "$SVPID" 2>/dev/null; continue; fi

  # fire several CONCURRENT streaming generations — more in-flight connection
  # threads = higher chance one is inside complete() exactly when deinit runs.
  CURLPIDS=()
  for _ in 1 2 3; do
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"mlx-serve","stream":true,"max_tokens":400,"temperature":0.7,"messages":[{"role":"user","content":"Write a long detailed essay about the history of computing, at least 300 words."}]}' \
      >/dev/null 2>&1 &
    CURLPIDS+=($!)
  done

  # land SIGTERM at a varied phase (prefill vs mid-decode)
  case $((i % 4)) in 0) d=0.4;; 1) d=0.8;; 2) d=1.4;; 3) d=2.2;; esac
  sleep "$d"
  kill -TERM "$SVPID" 2>/dev/null

  # wait for exit within the drain bound (+margin); detect hang
  exited=0
  for _ in $(seq 1 70); do   # 35s
    kill -0 "$SVPID" 2>/dev/null || { exited=1; break; }
    sleep 0.5
  done
  for cp in "${CURLPIDS[@]}"; do kill -9 "$cp" 2>/dev/null; done
  if [ "$exited" != 1 ]; then
    echo -e "  ${RED}FAIL${NC} iter $i (delay ${d}s): server HUNG after SIGTERM"
    kill -9 "$SVPID" 2>/dev/null; FAIL=$((FAIL+1)); continue
  fi
  wait "$SVPID" 2>/dev/null; rc=$?
  # 139 = 128+11 SIGSEGV ; 134 = SIGABRT. Clean graceful exit is 0.
  if [ "$rc" = 139 ] || [ "$rc" = 134 ]; then
    echo -e "  ${RED}FAIL${NC} iter $i (delay ${d}s): crashed rc=$rc"
    FAIL=$((FAIL+1))
  else
    echo -e "  ${GREEN}PASS${NC} iter $i (delay ${d}s): clean exit rc=$rc"
    PASS=$((PASS+1))
  fi
done

echo "=== shutdown-midstream: $PASS passed, $FAIL failed ($ITERS iters) ==="
[ "$FAIL" -eq 0 ]
