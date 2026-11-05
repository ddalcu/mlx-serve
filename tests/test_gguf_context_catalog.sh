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
python3 - "$WORK/models/org" <<'PY'
import json, pathlib, struct, sys
def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data
for name, context, configured in [('trained', 131072, None),
                                  ('zero', 0, 24576), ('missing', None, 24576)]:
    model = pathlib.Path(sys.argv[1]) / name
    model.mkdir(parents=True, exist_ok=True)
    header = b'GGUF' + struct.pack('<IQQ', 3, 0, 1 + (context is not None))
    header += string('general.architecture') + struct.pack('<I', 8) + string('glm-dsa')
    if context is not None:
        header += string('glm-dsa.context_length') + struct.pack('<IQ', 10, context)
    (model / 'model.gguf').write_bytes(header)
    if configured is not None:
        (model / 'config.json').write_text(json.dumps({'model_type': 'llama',
                                                      'max_position_embeddings': configured}))
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
for name, expected in [('trained', 131072), ('zero', 24576), ('missing', 24576)]:
    row = next(row for row in rows if row['id'] == 'org/' + name)
    assert row['loaded'] is False, row
    assert row['context_length'] == row['max_model_len'] == expected, row
    assert row['meta']['context_length'] == row['agent_budget']['context'] == expected, row
print('PASS: unloaded GGUF context and config-only context agree at every client surface')
PY
