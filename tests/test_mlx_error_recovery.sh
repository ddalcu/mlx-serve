#!/bin/bash
# An MLX error costs ONE REQUEST, never the server (#353). mlx-c's default handler was
# exit(-1). `MLX_SERVE_MLX_FAULT_CHUNK=<n>` / `_STEP=<n>` latch a synthetic Metal OOM at the
# n-th prefill-chunk / decode-step checkpoint and disarm, so one boot exercises the failure
# (a 503 with a body, `[mlx]` in the log) and the recovery (200 on the next request).
#
# Usage: ./tests/test_mlx_error_recovery.sh [model_dir] [port]
set -u
MODEL=${1:-"$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"}
PORT=${2:-8151}
BASE="http://127.0.0.1:$PORT"
PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

[ -d "$MODEL" ] || { echo "SKIP: model not found at $MODEL"; exit 0; }
[ -x ./zig-out/bin/mlx-serve ] || { echo "FAIL: build with -Doptimize=ReleaseFast first"; exit 1; }

LOG=$(mktemp -t mlxerr).log
# Chunk 2, not 1, so real forward work precedes the fault.
MLX_SERVE_MLX_FAULT_CHUNK=2 ./zig-out/bin/mlx-serve serve --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done

req() { # $1 = prompt
  curl -s -o /tmp/mlxerr_body.json -w '%{http_code}' -m 300 \
    -H 'content-type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":8,\"temperature\":0,\"stream\":false}" \
    "$BASE/v1/chat/completions"
}

echo "[1] the request that hits the injected MLX error is refused, by name"
# Long enough to prefill in more than one chunk (the default chunk is 512 at
# its narrowest), so the fault lands after real forward work.
FAULT_PROMPT=$(python3 -c "print('Explain the following list. ' + ' '.join(str(i) for i in range(4000)))")
CODE=$(req "$FAULT_PROMPT")
# 503 ONLY: the memory classification is part of what is under test, and a body is as
# load-bearing as the code.
case "$CODE" in
  503) ok "injected MLX OOM answered 503 ($(head -c 120 /tmp/mlxerr_body.json))" ;;
  000|"") bad "no HTTP response — the server died (the #353 symptom)" ;;
  500) bad "answered 500: the memory class was lost between the latch and the surface" ;;
  *)   bad "unexpected status $CODE: $(head -c 200 /tmp/mlxerr_body.json)" ;;
esac
grep -q "FAULT INJECTION armed" "$LOG" || bad "injector never armed — the env hook moved"
grep -q "\[mlx\] " "$LOG" || bad "no [mlx] line logged for the latched error"

echo "[2] the server is still serving: the NEXT request succeeds"
kill -0 $SRV 2>/dev/null || bad "server process is gone"
CODE2=$(req "Say hello.")
if [ "$CODE2" = "200" ] && grep -q '"content"' /tmp/mlxerr_body.json; then
  ok "second request answered 200 with content"
else
  bad "second request status $CODE2: $(head -c 200 /tmp/mlxerr_body.json)"
fi

echo "[3] the fault is ONE-SHOT: a third request is unaffected"
CODE3=$(req "Say goodbye.")
[ "$CODE3" = "200" ] && ok "third request answered 200" || bad "third request status $CODE3"

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null

# The decode checkpoint: a decode-time failure used to finish 200 and become the next request's 503.
echo "[4] an MLX error during DECODE fails that request, not the next one"
LOG2=$(mktemp -t mlxerr2).log
MLX_SERVE_MLX_FAULT_STEP=2 ./zig-out/bin/mlx-serve serve --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG2" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
CODE4=$(req "Write one sentence about the sea.")
case "$CODE4" in
  503|500) ok "decode-time MLX error answered $CODE4" ;;
  000|"")  bad "no HTTP response — the server died during decode" ;;
  200)     bad "decode-time MLX error was SWALLOWED: request answered 200 (the defect)" ;;
  *)       bad "unexpected status $CODE4: $(head -c 200 /tmp/mlxerr_body.json)" ;;
esac
kill -0 $SRV 2>/dev/null || bad "server process is gone after the decode fault"
CODE5=$(req "Say hello.")
if [ "$CODE5" = "200" ] && grep -q '"content"' /tmp/mlxerr_body.json; then
  ok "the request after a decode fault answered 200 with content"
else
  bad "post-decode-fault request status $CODE5: $(head -c 200 /tmp/mlxerr_body.json)"
fi

# The streaming half: the SSE head is already on the wire, so the bar is the terminal `error` event.
echo "[6] a decode-time MLX error on a STREAMING request sends the mapped SSE error"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
LOG3=$(mktemp -t mlxerr3).log
MLX_SERVE_MLX_FAULT_STEP=2 ./zig-out/bin/mlx-serve serve --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG3" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s -m 300 -N -H 'content-type: application/json' \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Write one sentence about the sea."}],"max_tokens":32,"temperature":0,"stream":true}' \
  "$BASE/v1/chat/completions" > /tmp/mlxerr_stream.txt 2>&1
if grep -q '"finish_reason":"error"' /tmp/mlxerr_stream.txt &&
   grep -q 'ran out of GPU memory' /tmp/mlxerr_stream.txt &&
   grep -q '"code":503' /tmp/mlxerr_stream.txt; then
  ok "streaming decode fault sent the mapped 503 error event"
else
  bad "streaming decode fault frame: $(tail -c 300 /tmp/mlxerr_stream.txt)"
fi
# The pre-review shape: a raw Zig error name shipped to the client.
grep -q 'Internal server error: GenerationOutOfMemory' /tmp/mlxerr_stream.txt &&
  bad "the raw error NAME is still what the streaming client is told"
# And the same server still serves.
CODE7=$(req "Say hello.")
[ "$CODE7" = "200" ] && ok "the request after a streaming decode fault answered 200" ||
  bad "post-streaming-fault request status $CODE7"

# A long prompt with NO max_tokens field: the omitted sentinel (maxInt/4) must be clamped before the guard bills it.
echo "[5] a long prompt with NO max_tokens field is admitted"
# Needs a prompt past KVCache.RESERVE_MIN_TOKENS (32k) that still fits the context.
CTX=$(curl -s -m 10 "$BASE/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); print(max([m.get('context_length',0) for m in d.get('data',[])] or [0]))" 2>/dev/null || echo 0)
if [ "${CTX:-0}" -lt 65536 ]; then
  echo "  SKIP: model context $CTX < 65536 — no room for a 45k-token prompt"
else
LONG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 9000)[:180000])")
CODE6=$(python3 - "$BASE" "$LONG" <<'PY2'
import json,sys,urllib.request,urllib.error
base,long_text=sys.argv[1:3]
body={"model":"mlx-serve","messages":[{"role":"user","content":long_text+"\n\nReply with one word."}],
      "temperature":0,"stream":False}   # NO max_tokens field on purpose
req=urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(body).encode(),
                           headers={"content-type":"application/json"})
try:
    with urllib.request.urlopen(req,timeout=1800) as r:
        json.load(r); print("200")
except urllib.error.HTTPError as e:
    open("/tmp/mlxerr_body.json","w").write(e.read().decode("utf-8","replace")); print(e.code)
except Exception as e:
    open("/tmp/mlxerr_body.json","w").write(f"{type(e).__name__}: {e}"); print("000")
PY2
)
if [ "$CODE6" = "200" ]; then
  ok "omitted max_tokens on a long prompt answered 200"
else
  bad "omitted max_tokens refused with $CODE6: $(head -c 240 /tmp/mlxerr_body.json)"
fi
fi

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "---- $PASS passed, $FAIL failed ----"
[ "$FAIL" -eq 0 ]
