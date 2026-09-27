#!/usr/bin/env bash
# `/api/pull org/name` registers the model under `org/name`, the id discovery
# gives the same directory — not its bare basename (#450).
#
# FULLY HERMETIC: HOME points at a temp dir that already holds the "pulled"
# model, so the pull takes the on-disk fast path and never touches the network.
# The model-dir root is empty, so discovery cannot register the path first.
#
# Usage: ./tests/test_pull_registers_org_id.sh [port]
set -uo pipefail
PORT="${1:-11379}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-pull.XXXXXX")"
SRV=""
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT

MODEL="$TMP/home/.mlx-serve/models/org/pulled"
mkdir -p "$MODEL" "$TMP/empty"
printf '{"model_type":"llama"}' > "$MODEL/config.json"
printf '0123' > "$MODEL/model.safetensors"

HOME="$TMP/home" "$BIN" --serve --host 127.0.0.1 --port "$PORT" --model-dir "$TMP/empty" >"$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$TMP/server.log"; exit 1; }
  sleep 0.5
done

rc=0
pull=$(curl -s -X POST "http://127.0.0.1:$PORT/api/pull" -H 'Content-Type: application/json' \
  -d '{"model":"org/pulled","stream":false}')
echo "$pull" | grep -q '"status":"success"' && echo "PASS: pull fast path" || { echo "FAIL: pull ($pull)"; rc=1; }

ids=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))')
case " $ids " in
  *" org/pulled "*) echo "PASS: pulled model listed as org/pulled" ;;
  *) echo "FAIL: pulled model not listed as org/pulled (ids: $ids)"; rc=1 ;;
esac
case " $ids " in
  *" pulled "*) echo "FAIL: pulled model also listed under its bare basename"; rc=1 ;;
esac

[ $rc -eq 0 ] && echo "ALL PASS" || echo "FAILED"
exit $rc
