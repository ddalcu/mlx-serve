#!/usr/bin/env bash
# The plugin kit's proof (plan 3.10): the kit staged by `zig build kit` builds `_example` (and a `plugin new`
# copy of it) as dylibs with no checkout of the server; `mlx-serve plugin check|fixture|install` take them; and the
# built binary, with plugins switched on in a scratch HOME, loads `_example` and serves its tiny model (the
# plugin's own fixture written as a pack) on /v1/models and /v1/chat/completions. With `--no-plugins` discovery
# skips the pack. No model download: the pack is the plugin's. Scratch under ~/claude-tmp.
#
# Usage: ./tests/test_plugin_kit.sh [port]   (build first: zig build -Doptimize=ReleaseFast)
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11391}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
ZIG="$ROOT/.zig-toolchain/zig"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
[ -x "$ZIG" ] || { echo "FAIL: stage the pinned Zig first (scripts/fetch-zig.sh)"; exit 1; }

PASS=0; FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $1"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

mkdir -p "$HOME/claude-tmp"
SCRATCH="$(mktemp -d "$HOME/claude-tmp/plugin-kit.XXXXXX")"
FAKE_HOME="$SCRATCH/home"
mkdir -p "$FAKE_HOME/.mlx-serve"
LOG="$SCRATCH/server.log"
SRV=""
cleanup() {
    srv_stop "$SRV"
    rm -rf "$SCRATCH"
}
trap cleanup EXIT
mlxs() { HOME="$FAKE_HOME" "$BIN" "$@"; }

echo "[1] the kit stages and builds its example from itself"
(cd "$ROOT" && "$ZIG" build kit) >"$SCRATCH/kit.log" 2>&1
KIT="$(ls -d "$ROOT"/zig-out/kit/mlx-serve-plugin-kit-*/ 2>/dev/null | head -1)"
check "zig build kit staged a kit" "$([ -n "$KIT" ] && [ -f "$KIT/build.zig" ] && echo 1 || echo 0)"
check "the kit carries the C header" "$([ -f "$KIT/include/mlx_serve_plugin.h" ] && echo 1 || echo 0)"
cp -R "$KIT" "$SCRATCH/kit"
KIT="$SCRATCH/kit"
(cd "$KIT/examples/_example" && "$ZIG" build --prefix "$SCRATCH/out/_example" --cache-dir "$SCRATCH/zig-cache") >"$SCRATCH/build.log" 2>&1
DYLIB="$SCRATCH/out/_example/lib/lib_example.dylib"
check "_example builds from the kit alone" "$([ -f "$DYLIB" ] && echo 1 || echo 0)"
check "the dylib links no MLX of its own" "$(otool -L "$DYLIB" | grep -q mlx && echo 0 || echo 1)"

echo "[2] plugin new | check | fixture | install"
mlxs plugin new toy --kit "$KIT" --dir "$SCRATCH" >/dev/null 2>&1
(cd "$SCRATCH/toy" && "$ZIG" build --cache-dir "$SCRATCH/zig-cache") >>"$SCRATCH/build.log" 2>&1
check "plugin new toy builds libtoy.dylib" "$([ -f "$SCRATCH/toy/zig-out/lib/libtoy.dylib" ] && echo 1 || echo 0)"
check "plugin check passes the new plugin" "$(mlxs plugin check "$SCRATCH/toy" >/dev/null 2>&1 && echo 1 || echo 0)"
PACK="$SCRATCH/models/t/example"
mlxs plugin fixture "$DYLIB" "$PACK" >/dev/null 2>&1
check "plugin fixture writes a pack" "$([ -f "$PACK/config.json" ] && [ -f "$PACK/tokenizer.json" ] && echo 1 || echo 0)"
OUT="$(mlxs plugin check "$DYLIB" --model "$PACK" 2>&1)"
check "plugin check --model: conformant, claims and parses the pack" "$(echo "$OUT" | grep -q 'claims and parses' && echo 1 || echo 0)"
mlxs plugin install "$DYLIB" >/dev/null 2>&1
RECEIPT="$FAKE_HOME/.mlx-serve/plugins/_example.json"
SHA="$(shasum -a 256 "$FAKE_HOME/.mlx-serve/plugins/_example.dylib" 2>/dev/null | cut -d' ' -f1)"
check "the receipt records the file's hash" "$(grep -q "\"sha256\":\"$SHA\"" "$RECEIPT" 2>/dev/null && echo 1 || echo 0)"
check "the receipt records version and signer" "$(grep -q '"version":"0.1.0"' "$RECEIPT" && grep -q '"signature":"' "$RECEIPT" && echo 1 || echo 0)"
xattr -w com.apple.quarantine "0081;00000000;Safari;" "$SCRATCH/toy/zig-out/lib/libtoy.dylib"
check "a quarantined dylib is not installed unconfirmed" "$(mlxs plugin install "$SCRATCH/toy/zig-out/lib/libtoy.dylib" </dev/null >/dev/null 2>&1 && echo 0 || echo 1)"
check "... and leaves nothing in the plugins folder" "$([ ! -e "$FAKE_HOME/.mlx-serve/plugins/toy.dylib" ] && echo 1 || echo 0)"

URL="http://127.0.0.1:$PORT"
serve() {
    HOME="$FAKE_HOME" "$BIN" --serve --model-dir "$SCRATCH/models" --port "$PORT" --log-level debug --log-file off "$@" >"$LOG" 2>&1 &
    SRV=$!
}
row() { curl -s "$URL/v1/models" | python3 -c "import json,sys; r=[m for m in json.load(sys.stdin)['data'] if m['id']==sys.argv[1]]; print(json.dumps(r[0]) if r else '')" "$1"; }
field() { python3 -c "import json,sys; d=json.loads(sys.argv[1] or 'null'); print((d or {}).get(sys.argv[2], ''))" "$1" "$2"; }

echo "[3] the built binary loads _example and serves its pack"
echo '{"enabled": true}' >"$FAKE_HOME/.mlx-serve/plugins.json"
serve --model "$PACK"
if srv_wait_health "$URL" 120 0.5 "$SRV"; then
    check "startup logs the plugin loaded" "$(grep -q '\[plugins\] loaded _example' "$LOG" && echo 1 || echo 0)"
    R="$(row t/example)"
    check "/v1/models lists the pack with the plugin badge" "$([ "$(field "$R" plugin)" = "_example" ] && echo 1 || echo 0)"
    CODE=$(curl -s -o "$SCRATCH/chat.json" -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"t/example","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}')
    check "chat on the plugin's model → 200 ($CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
    TOKS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['usage']['completion_tokens'])" "$SCRATCH/chat.json" 2>/dev/null || echo 0)
    check "it generated tokens ($TOKS)" "$([ "${TOKS:-0}" -gt 0 ] && echo 1 || echo 0)"
    STREAM=$(curl -sN "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"t/example","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"stream":true}')
    check "a streamed chat carries content and ends with [DONE]" "$(echo "$STREAM" | grep -q '"content":"[^"]' && echo "$STREAM" | grep -q '\[DONE\]' && ! echo "$STREAM" | grep -q '"error"' && echo 1 || echo 0)"
else
    check "server with plugins on became healthy" 0
    tail -20 "$LOG"
fi
srv_stop "$SRV"; SRV=""

echo "[4] --no-plugins: discovery skips the pack nobody claims"
serve --no-plugins
if srv_wait_health "$URL" 120 0.5 "$SRV"; then
    check "no plugin is loaded" "$(grep -q '\[plugins\] loaded' "$LOG" && echo 0 || echo 1)"
    check "the pack has no row" "$([ -z "$(row t/example)" ] && echo 1 || echo 0)"
else
    check "server with plugins off became healthy" 0
    tail -20 "$LOG"
fi

echo
echo "plugin kit: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
