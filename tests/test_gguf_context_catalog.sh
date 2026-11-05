#!/usr/bin/env bash
# A header-only GGUF advertises its trained context before the first model load.
set -euo pipefail
source "$(dirname "$0")/lib/server.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
PORT="${1:-11396}"
WORK="$(mktemp -d)"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT
[ -x "$BIN" ] || { echo "FAIL: build mlx-serve first"; exit 1; }
mkdir -p "$WORK/models/org/trained"
python3 - "$WORK/models/org/trained/model.gguf" <<'PY'
import pathlib, struct, sys
def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data
header = b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
header += string('general.architecture') + struct.pack('<I', 8) + string('glm-dsa')
header += string('glm-dsa.context_length') + struct.pack('<IQ', 10, 131072)
pathlib.Path(sys.argv[1]).write_bytes(header)
PY
HOME="$WORK" "$BIN" --serve --model-dir "$WORK/models" --host 127.0.0.1 \
    --port "$PORT" --log-file off >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
if ! srv_wait_health "http://127.0.0.1:$PORT" 60 0.2 "$SERVER_PID"; then
    cat "$WORK/server.log" >&2
    exit 1
fi
python3 - "$PORT" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models', timeout=10) as response:
    rows = json.load(response)['data']
row = next(row for row in rows if row['id'] == 'org/trained')
assert row['loaded'] is False, row
assert row['context_length'] == row['max_model_len'] == 131072, row
assert row['meta']['context_length'] == row['agent_budget']['context'] == 131072, row
print('PASS: unloaded GGUF catalog context agrees at every client surface')
PY
