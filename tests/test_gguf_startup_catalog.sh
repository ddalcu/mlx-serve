#!/usr/bin/env bash
# A server started on a GGUF (`--model x.gguf --model-dir root`) lists every discovered model, not only the GGUF,
# serves the GGUF under its discovered folder's id (no second row for the same file), still rescans its roots, and
# gates the siblings' loads by memory and leaves them their own context like every other boot.
#
# With SIBLING_MLX_MODEL (an MLX chat model dir) it also loads that model beside the GGUF and checks it runs with the
# settings a headless boot gives it (PLD, prefix cache, spec flags); with MEDIA_BOOT_MODEL (a media pack) the same
# beside a server started on that pack.
#
#   LLAMA_GGUF_MODEL=/path/to/model.gguf [SIBLING_MLX_MODEL=/path/to/mlx-model] [MEDIA_BOOT_MODEL=/path/to/pack] \
#       ./tests/test_gguf_startup_catalog.sh [port]   # LLAMA_GGUF_MODEL unset = SKIP
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
PORT="${1:-11412}"
MODEL="${LLAMA_GGUF_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP test_gguf_startup_catalog: set LLAMA_GGUF_MODEL=/path/to/model.gguf to run"
    exit 0
fi
[ -x "$BIN" ] || { echo "FAIL: build mlx-serve first"; exit 1; }
MODEL_ABS="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
WORK="$(mktemp -d)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && srv_stop "$SERVER_PID"
    rm -rf "$WORK"
}
trap cleanup EXIT

stub_mlx_model() {
    mkdir -p "$WORK/models/$1"
    echo '{"model_type":"llama","max_position_embeddings":32768}' > "$WORK/models/$1/config.json"
    head -c 2000000 /dev/zero > "$WORK/models/$1/model.safetensors"
}
mkdir -p "$WORK/home" "$WORK/models/testorg/pack-GGUF"
ln -s "$MODEL_ABS" "$WORK/models/testorg/pack-GGUF/model.gguf"
stub_mlx_model testorg/sibling

HOME="$WORK/home" "$BIN" --serve --model "$WORK/models/testorg/pack-GGUF/model.gguf" --model-dir "$WORK/models" \
    --host 127.0.0.1 --port "$PORT" --log-file off >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
if ! srv_wait_health "http://127.0.0.1:$PORT" 120 0.5 "$SERVER_PID"; then
    cat "$WORK/server.log" >&2
    exit 1
fi

stub_mlx_model testorg/late
curl -s -m 30 -X POST "http://127.0.0.1:$PORT/v1/models/rescan" >/dev/null

python3 -I - "$PORT" "$WORK/server.log" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models', timeout=30) as response:
    rows = {row['id']: row for row in json.load(response)['data']}
fails = 0
def check(desc, ok):
    global fails
    print(('PASS ' if ok else 'FAIL ') + desc)
    fails += 0 if ok else 1
check('the sibling MLX model is listed beside the GGUF', 'testorg/sibling' in rows)
check('the GGUF is served under its discovered folder id', rows.get('testorg/pack-GGUF', {}).get('loaded') is True)
check('the GGUF file has no second row of its own', 'model' not in rows)
check('the sibling keeps its own context, not the GGUF engine default', rows.get('testorg/sibling', {}).get('context_length') == 32768)
check('the GGUF keeps the llama.cpp default context', rows.get('testorg/pack-GGUF', {}).get('context_length') == 8192)
check('a model added after boot is found by /v1/models/rescan', 'testorg/late' in rows)
log = open(sys.argv[2]).read()
check('siblings load under the memory gate',
      '[registry] max_resident_models=' in log and 'max_resident_mem=unlimited' not in log)
sys.exit(1 if fails else 0)
PY
[ $? -eq 0 ] || exit 1

SIBLING="${SIBLING_MLX_MODEL:-}"
if [ -z "$SIBLING" ] || [ ! -d "$SIBLING" ]; then
    echo "SKIP sibling settings: set SIBLING_MLX_MODEL to run"
    exit 0
fi
SIBLING="$(cd "$SIBLING" && pwd)"
srv_stop "$SERVER_PID"; SERVER_PID=""
# The sibling's /props settings once it is the default, on a server started with the given extra args.
sibling_settings() {
    HOME="$WORK/home" "$BIN" --serve "$@" --model-dir "$WORK/models" --host 127.0.0.1 --port "$PORT" \
        --log-file off >"$WORK/sibling.log" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "http://127.0.0.1:$PORT" 120 0.5 "$SERVER_PID" || { cat "$WORK/sibling.log" >&2; return 1; }
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/load-model" -H 'content-type: application/json' \
        -d "{\"model\":\"$SIBLING\",\"default\":true}" >/dev/null
    curl -s -m 30 "http://127.0.0.1:$PORT/props" |
        python3 -I -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["settings"], sort_keys=True, indent=1))'
    srv_stop "$SERVER_PID"; SERVER_PID=""
}
ON_HEADLESS="$(sibling_settings)"
SIB_FAIL=0
same_as_headless() {
    if [ -n "$2" ] && [ "$2" = "$ON_HEADLESS" ]; then
        echo "PASS a model loaded beside $1 runs with the settings a headless boot gives it"
    else
        echo "FAIL a model loaded beside $1 runs with the settings a headless boot gives it"
        diff <(echo "$2") <(echo "$ON_HEADLESS")
        SIB_FAIL=1
    fi
}
same_as_headless "the GGUF" "$(sibling_settings --model "$WORK/models/testorg/pack-GGUF/model.gguf")"
MEDIA="${MEDIA_BOOT_MODEL:-}"
if [ -n "$MEDIA" ] && [ -d "$MEDIA" ]; then
    same_as_headless "a media pack" "$(sibling_settings --model "$(cd "$MEDIA" && pwd)")"
else
    echo "SKIP media boot: set MEDIA_BOOT_MODEL to run"
fi
exit "$SIB_FAIL"
