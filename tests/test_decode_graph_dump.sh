#!/bin/bash
# test_decode_graph_dump.sh — end-to-end guard for MLX_SERVE_DECODE_GRAPH_DUMP.
#
# Boots a real model with the decode forward micro-bench and the graph dump
# armed, then checks that the dump names the primitives of one decode forward.
#
# Artifacts, in $OUT (default /tmp/decode-graph-dump-$PORT):
#   decode-graph.txt          the lazy graph of one S=1 decode forward
#   decode-graph-summary.txt  node count and per-primitive counts, sorted —
#                             rerun on the same model and build and diff it
#   server.log                the boot log
#
# Usage: ./tests/test_decode_graph_dump.sh <model-dir> [extra mlx-serve flags...]

set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:?usage: test_decode_graph_dump.sh <model-dir> [flags...]}"
shift
PORT="${PORT:-8098}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"
OUT="${OUT:-/tmp/decode-graph-dump-$PORT}"
DUMP="$OUT/decode-graph.txt"
SUMMARY="$OUT/decode-graph-summary.txt"
LOG="$OUT/server.log"
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

mkdir -p "$OUT"
rm -f "$DUMP" "$SUMMARY"

MLX_SERVE_DECODE_GRAPH_DUMP="$DUMP" MLX_SERVE_DECODE_FWD_UBENCH=2 \
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info \
  --log-file off "$@" >"$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 900); do
  grep -q "\[fwd-ubench\] done" "$LOG" && break
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null

echo "decode graph dump: $MODEL"

if grep -q "\[fwd-ubench\] graph dumped to $DUMP" "$LOG"; then
  ok "server logged the dump path"
else
  fail "no '[fwd-ubench] graph dumped to $DUMP' line in $LOG"
fi

if grep -q "decode forwards, eval-per-step" "$LOG"; then
  ok "micro-bench still ran after the dump"
else
  fail "micro-bench result line missing from $LOG"
fi

if [ -s "$DUMP" ]; then
  ok "dump file written"
else
  fail "dump file missing or empty: $DUMP"
fi

if head -1 "$DUMP" 2>/dev/null | grep -q "^Inputs: "; then
  ok "dump starts with the graph's inputs"
else
  fail "first line of $DUMP is not 'Inputs: ...'"
fi

# A node line is "<Primitive> <inputs> -> <outputs>".
NODES=$(grep -c -- " -> " "$DUMP" 2>/dev/null)
NODES=${NODES:-0}
if [ "$NODES" -ge 10 ]; then
  ok "dump holds $NODES graph nodes"
else
  fail "dump holds $NODES graph nodes, expected a full forward (>= 10)"
fi

{
  echo "model: $MODEL"
  echo "nodes: $NODES"
  grep -- " -> " "$DUMP" 2>/dev/null | awk '{print $1}' | sort | uniq -c | sort -rn
} >"$SUMMARY"
echo "  artifacts: $DUMP, $SUMMARY"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
