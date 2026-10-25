#!/bin/bash
# A template that cannot keep a late system note in place warns once per template.
set -euo pipefail
MODEL="${1:?Usage: test_late_system_fold_warning.sh <model_dir whose template rejects late system turns> [port]}"
PORT="${2:-11369}"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
SCRATCH=$(mktemp -d)
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$SCRATCH"
}
trap cleanup EXIT
HOME="$SCRATCH" "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --ctx-size 4096 --no-mtp --no-pld --no-drafter \
    --log-file "$SCRATCH/server.log" > "$SCRATCH/stdout.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        printf 'FAIL: server stopped during load\n'
        exit 1
    fi
    sleep 1
done
post() {
    curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$1" > /dev/null
}
warnings() {
    grep -c 'cannot keep a late system message in place' "$SCRATCH/server.log" || true
}

post '{"model":"mlx-serve","max_tokens":4,"temperature":0,"messages":[{"role":"system","content":"Be brief."},{"role":"system","content":"Hook."},{"role":"user","content":"Name a color."}]}'
if [ "$(warnings)" != 0 ]; then
    printf 'FAIL: a leading system run warned\n'
    exit 1
fi
printf 'PASS: leading system run does not warn\n'

for note in "Reminder one." "Reminder two."; do
    post "{\"model\":\"mlx-serve\",\"max_tokens\":4,\"temperature\":0,\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Name a color.\"},{\"role\":\"system\",\"content\":\"$note\"}]}"
done
count=$(warnings)
if [ "$count" != 1 ]; then
    printf 'FAIL: expected one late-system fold warning after two late notes, got %s\n' "$count"
    exit 1
fi
if grep -q 'jinja render failed' "$SCRATCH/server.log"; then
    printf 'FAIL: native template fell back\n'
    exit 1
fi
printf 'PASS: late system notes warn once per template\n'
