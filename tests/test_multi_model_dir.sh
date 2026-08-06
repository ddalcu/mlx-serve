#!/usr/bin/env bash
# `--model-dir` is REPEATABLE, and every folder it names is actually served.
#
# Before this the flag took ONE directory, so a user whose models lived in more
# than one place (the app's download folder, the folder it used to be, an
# LM Studio tree) had the rest listed by the app's own picker and absent from
# /v1/models — switching to one cost a full server restart instead of a hot
# swap, because the id was not in the registry.
#
# FULLY HERMETIC: the "models" are empty dirs holding a config.json. Nothing is
# ever loaded — this pins DISCOVERY and the arg loop, which is where the bug was.
#
# Usage: ./tests/test_multi_model_dir.sh [port]
set -uo pipefail
PORT="${1:-11378}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-roots.XXXXXX")"
SRV=""
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT

mk() { mkdir -p "$TMP/$1"; printf '{"model_type":"%s"}' "$2" > "$TMP/$1/config.json"; }
mk a/org/alpha  qwen3
mk a/org/shared qwen3
mk b/org/beta   llama
mk b/org/shared mistral     # same id as a/org/shared — first root must win
mk c/org/gamma  gemma3
rc=0

boot() { # boot <logfile> <dir...>
  local log="$1"; shift
  local args=()
  for d in "$@"; do args+=(--model-dir "$d"); done
  "$BIN" --serve --port "$PORT" "${args[@]}" >"$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$log"; return 1; }
    sleep 0.5
  done
  echo "FAIL: server never became healthy"; return 1
}
stop() { [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; SRV=""; }
ids() { curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c \
  "import sys,json;print(' '.join(sorted(m['id'] for m in json.load(sys.stdin)['data'])))"; }

# ── [1] Three folders, all served, duplicate id resolved to the FIRST.
boot "$TMP/multi.log" "$TMP/a" "$TMP/b" "$TMP/c" || exit 1
GOT="$(ids)"
WANT="org/alpha org/beta org/gamma org/shared"
if [ "$GOT" = "$WANT" ]; then
  echo "PASS: every --model-dir is discovered ($GOT)"
else
  echo "FAIL: expected [$WANT], got [$GOT]"; rc=1
fi
# The surviving `org/shared` must be the FIRST root's copy — the app orders the
# download destination first precisely so its copy is the live one. The two
# copies declare different architectures, so the served entry names which folder
# it came from; a count of one would pass either way.
ARCH=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c \
  "import sys,json
d=[m for m in json.load(sys.stdin)['data'] if m['id']=='org/shared']
print(d[0].get('meta',{}).get('architecture','') if d else '')")
[ "$ARCH" = "qwen3" ] \
  && echo "PASS: a duplicate id resolves to the FIRST folder (arch $ARCH)" \
  || { echo "FAIL: duplicate resolved to arch '$ARCH', wanted the first folder's qwen3"; rc=1; }
grep -q "already found in an earlier --model-dir" "$TMP/multi.log" \
  && echo "PASS: the skipped duplicate is logged" \
  || { echo "FAIL: nothing logged about the skipped duplicate"; rc=1; }
stop

# ── [2] One folder behaves exactly as it always did.
boot "$TMP/single.log" "$TMP/a" || exit 1
GOT="$(ids)"
[ "$GOT" = "org/alpha org/shared" ] \
  && echo "PASS: a single --model-dir is unchanged ($GOT)" \
  || { echo "FAIL: single-dir case returned [$GOT]"; rc=1; }
stop

# ── [3] An unreachable folder is SKIPPED, not fatal: the second folder can live
# on an external drive, and unplugging it must not stop the server.
boot "$TMP/missing.log" "$TMP/a" "$TMP/definitely-not-here" "$TMP/c" || exit 1
GOT="$(ids)"
[ "$GOT" = "org/alpha org/gamma org/shared" ] \
  && echo "PASS: an unreachable folder is skipped, the rest still serve" \
  || { echo "FAIL: with one bad folder got [$GOT]"; rc=1; }
grep -q "scan failed" "$TMP/missing.log" \
  && echo "PASS: the skipped folder is named in the log" \
  || { echo "FAIL: nothing logged about the unreachable folder"; rc=1; }
stop

# ── [4] Past the cap the server REFUSES rather than silently dropping folders —
# a launcher that ignores what it was asked to scan is worse than one that
# won't start (the silent-flag-eater class).
ARGS=()
for i in $(seq 1 9); do mkdir -p "$TMP/many$i"; ARGS+=(--model-dir "$TMP/many$i"); done
"$BIN" --serve --port "$PORT" "${ARGS[@]}" >"$TMP/toomany.log" 2>&1
if [ $? -ne 0 ] && grep -q "at most 8 folders" "$TMP/toomany.log"; then
  echo "PASS: a 9th --model-dir is refused by name"
else
  echo "FAIL: over-cap launch did not refuse (see $TMP/toomany.log)"; rc=1
fi

[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILURES"
exit $rc
