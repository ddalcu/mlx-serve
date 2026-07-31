#!/bin/bash
# Multi-client stress test — the "several pi's + several chats + voice + image gen" mix.
#
# Simulates the load pattern of multiple concurrent agent CLIs (pi) plus app chat
# tabs plus voice (TTS) plus image generation, all against ONE server, across
# MULTIPLE models (two chat models = per-model batch groups + multi-model
# registry; Kokoro TTS + FLUX klein ride the gen_queue on the same inference
# thread). Video is deliberately excluded (heavy, minutes-long).
#
# What it asserts is CORRECTNESS UNDER LOAD, not latency: every stream
# terminates cleanly ([DONE], finish_reason), every media request returns real
# bytes, no request is lost, and the server is alive and functional afterwards.
# A media generation legitimately stalls chat decode (single GPU, accepted
# design) — client timeouts here are generous on purpose.
#
# Usage:
#   ./tests/test_multiclient_stress.sh [port]
# Env:
#   STRESS_CHAT_A   agent-workload chat model dir (default gemma-4-e4b-it-4bit)
#   STRESS_CHAT_B   plain-chat model dir, should differ from A (default gemma-4-e2b-it-4bit)
#   STRESS_TTS      TTS model dir (default Kokoro-82M-MLX-Serve; empty to skip leg)
#   STRESS_IMAGE    image model dir (default FLUX.2-klein-4B-mflux-4bit; empty to skip leg)
#   N_AGENTS=3 N_PLAIN=2 N_TTS=3 N_IMG=2 MAX_TOKENS=200 ROUNDS=1
#   STRESS_THINK=1   send enable_thinking on chat A (reasoning models)
#   STRESS_MTP=1     send enable_mtp on chat A (MoE default is off)
#   N_ABORT=0        clients that DISCONNECT mid-stream (the pi-stop pattern)
#   N_CHURN=0        load/unload cycles on chat B (the app's media load→gen→unload pattern)
#   SOAK_WAVES=1     repeat the whole concurrent mix N times against ONE server,
#                    sampling /props memory after each wave; with >= 3 waves the
#                    final active_bytes must stay within 1.5x of wave 2's
#                    (steady state after warm-up — catches the leak class)
#   STRESS_CHAT_A2   alternate agent-workload model on even waves (soak realism)

set -u

PORT=${1:-8135}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

MODELS_ROOT="$HOME/.mlx-serve/models"
CHAT_A="${STRESS_CHAT_A:-$MODELS_ROOT/mlx-community/gemma-4-e4b-it-4bit}"
CHAT_B="${STRESS_CHAT_B:-$MODELS_ROOT/mlx-community/gemma-4-e2b-it-4bit}"
TTS="${STRESS_TTS-$MODELS_ROOT/ddalcu/Kokoro-82M-MLX-Serve}"
IMAGE="${STRESS_IMAGE-$MODELS_ROOT/Runpod/FLUX.2-klein-4B-mflux-4bit}"
N_AGENTS=${N_AGENTS:-3}
N_PLAIN=${N_PLAIN:-2}
N_TTS=${N_TTS:-3}
N_IMG=${N_IMG:-2}
N_ABORT=${N_ABORT:-0}
N_CHURN=${N_CHURN:-0}
MAX_TOKENS=${MAX_TOKENS:-200}
ROUNDS=${ROUNDS:-1}
export STRESS_THINK="${STRESS_THINK:-0}" STRESS_MTP="${STRESS_MTP:-0}"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
[ -x "$BINARY" ] || { echo -e "${RED}FAIL${NC} $BINARY not found (zig build -Doptimize=ReleaseFast)"; exit 1; }
[ -d "$CHAT_A" ] || { echo -e "${YELLOW}SKIP${NC} chat model A not found: $CHAT_A (set STRESS_CHAT_A)"; exit 0; }
[ -d "$CHAT_B" ] || { echo -e "${YELLOW}SKIP${NC} chat model B not found: $CHAT_B (set STRESS_CHAT_B)"; exit 0; }
[ -n "$TTS" ] && [ ! -d "$TTS" ] && { echo "note: TTS model missing, skipping TTS leg"; TTS=""; }
[ -n "$IMAGE" ] && [ ! -d "$IMAGE" ] && { echo "note: image model missing, skipping image leg"; IMAGE=""; }

if curl -sf "$BASE/health" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC} port $PORT already serving — pick another port"; exit 1
fi

TMP=$(mktemp -d /tmp/mlx-stress.XXXXXX)
mkdir -p "$TMP/results"
LOG="$TMP/server.log"
SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# ---------------------------------------------------------------- worker ----
cat > "$TMP/worker.py" <<'PY'
"""One stress worker. Argv: kind label base model extra...
kind: agent | plain | tts | img
Writes PASS/FAIL + detail to stdout; exit 0 on pass."""
import json, os, sys, time, urllib.request, urllib.error

kind, label, base, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MAX_TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 200
ROUNDS = int(sys.argv[6]) if len(sys.argv) > 6 else 1
TIMEOUT = 900  # generous: cold model loads + image gen block decode
THINK = os.environ.get("STRESS_THINK") == "1"
MTP = os.environ.get("STRESS_MTP") == "1"

def post(path, body, raw=False):
    """POST with bounded 503 retry. A 503 here is the server's HONEST
    backpressure — resident-model cap with every victim refcounted by an
    in-flight stream ("retry after current requests complete") or a full
    submit queue — and real clients (openai SDK, undici) retry it. The
    stress assertion is "no request LOST", not "no request ever queued"."""
    data = json.dumps(body).encode()
    for attempt in range(7):
        req = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=TIMEOUT)
            return resp if raw else resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < 6:
                time.sleep(min(2 * (attempt + 1), 8))
                continue
            raise

def stream_chat(messages, tools=None, abort_after=None):
    """Returns (content, tool_calls, finish_reason, saw_done).
    abort_after=N: close the connection after N data events (client vanish)."""
    body = {"model": model, "messages": messages, "stream": True,
            "max_tokens": MAX_TOKENS, "temperature": 0.2,
            "stream_options": {"include_usage": True}}
    if tools: body["tools"] = tools
    # Thinking/MTP ride only the AGENT workload (chat A is the reasoning
    # model). A thinking-enabled small model on the plain leg spends the whole
    # token budget inside <think> and "empty content" is model behavior, not a
    # server bug (measured: gemma-e2b, 250 tokens, two-line-poem prompt).
    if THINK and tools: body["enable_thinking"] = True
    if MTP and tools: body["enable_mtp"] = True
    resp = post("/v1/chat/completions", body, raw=True)
    if abort_after is not None:
        seen = 0
        for raw in resp:
            if raw.startswith(b"data:"): seen += 1
            if seen >= abort_after:
                resp.close()  # vanish mid-stream, like a killed pi
                return "", [], None, False
        return "", [], None, True
    content, tool_calls, finish, saw_done = [], [], None, False
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue  # SSE comments / keepalives / blank
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            break
        d = json.loads(payload)
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            if delta.get("content"): content.append(delta["content"])
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_calls.append({"id": tc.get("id", "call_0"),
                                       "name": fn.get("name", ""),
                                       "arguments": fn.get("arguments", "{}")})
            if ch.get("finish_reason"): finish = ch["finish_reason"]
    return "".join(content), tool_calls, finish, saw_done

SHELL_TOOL = [{"type": "function", "function": {
    "name": "shell", "description": "Run a shell command and return its output",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]

def run_agent():
    """pi-style: tool round-trip(s), transport must stay clean throughout."""
    for r in range(ROUNDS):
        msgs = [{"role": "system", "content": "You are a coding agent. Use the shell tool for any filesystem question."},
                {"role": "user", "content": f"[job {label} round {r}] List the files in the current directory with the shell tool, then tell me how many there are."}]
        content, tcs, finish, done = stream_chat(msgs, tools=SHELL_TOOL)
        assert done, f"round {r}: no [DONE]"
        if tcs:
            assert finish == "tool_calls", f"round {r}: tool calls but finish={finish}"
            for t in tcs:
                json.loads(t["arguments"])  # emitted args must be valid JSON (server invariant)
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": t["id"], "type": "function",
                 "function": {"name": t["name"], "arguments": t["arguments"]}} for t in tcs]})
            for t in tcs:
                msgs.append({"role": "tool", "tool_call_id": t["id"],
                             "content": "README.md\nsrc\ntests\nbuild.zig\n"})
            content2, tcs2, finish2, done2 = stream_chat(msgs, tools=SHELL_TOOL)
            assert done2, f"round {r}: no [DONE] on tool-result round"
            assert content2 or tcs2, f"round {r}: empty follow-up"
        else:
            assert content.strip(), f"round {r}: no tool call AND no content"
    return "agent rounds clean"

def run_plain():
    for r in range(ROUNDS):
        content, _, finish, done = stream_chat(
            [{"role": "user", "content": f"[job {label} round {r}] Write a two-line poem about the number {r + 7}."}])
        assert done, f"round {r}: no [DONE]"
        assert content.strip(), f"round {r}: empty content"
    return "plain rounds clean"

def run_tts():
    for r in range(ROUNDS):
        wav = post("/v1/audio/speech",
                   {"model": model, "input": f"Stress test utterance number {r} from worker {label}."})
        assert len(wav) > 10000, f"round {r}: tiny audio ({len(wav)} bytes)"
        assert wav[:4] == b"RIFF", f"round {r}: not a WAV"
    return "tts rounds clean"

def run_img():
    for r in range(ROUNDS):
        out = json.loads(post("/v1/images/generations",
                              {"model": model, "prompt": f"a small {['red', 'blue', 'green'][r % 3]} triangle on white",
                               "steps": 4, "size": "512x512"}))
        import base64
        png = base64.b64decode(out["data"][0]["b64_json"])
        assert png[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]), f"round {r}: not a PNG"
    return "img rounds clean"

def run_abort():
    """Start streams and vanish mid-generation — the killed-pi / closed-tab
    pattern. Server must cancel the slot and stay healthy."""
    for r in range(ROUNDS):
        stream_chat([{"role": "user", "content":
                      f"[job {label} round {r}] Count from 1 to 500, one number per line."}],
                    abort_after=3)
        time.sleep(0.5)
    return "aborted streams issued"

def run_churn():
    """The app's media flow: load -> use -> unload, repeatedly, while chat
    traffic runs against OTHER models."""
    for r in range(ROUNDS):
        post("/v1/load-model", {"model": model})
        content, _, _, done = stream_chat(
            [{"role": "user", "content": f"[job {label} round {r}] Say the word ping."}])
        assert done, f"round {r}: no [DONE] on churn chat"
        post("/v1/unload-model", {"model": model})
    return "load/unload churn clean"

t0 = time.time()
try:
    detail = {"agent": run_agent, "plain": run_plain, "tts": run_tts, "img": run_img,
              "abort": run_abort, "churn": run_churn}[kind]()
    print(f"PASS {label}: {detail} ({time.time() - t0:.1f}s)")
except Exception as e:
    print(f"FAIL {label}: {type(e).__name__}: {e} ({time.time() - t0:.1f}s)")
    sys.exit(1)
PY

# ---------------------------------------------------------------- server ----
echo "== booting headless server on :$PORT (log: $LOG) =="
"$BINARY" --serve --port "$PORT" --model-dir "$MODELS_ROOT" \
    --max-concurrent 8 --metrics --log-level info --log-file "$LOG" > "$TMP/stdout.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 0.5; done
curl -sf "$BASE/health" >/dev/null || { echo -e "${RED}FAIL${NC} server did not come up"; tail -20 "$TMP/stdout.log"; exit 1; }

resolve_id() {  # dir -> served model id (matched on basename)
    curl -s "$BASE/v1/models" | python3 -c "
import sys, json, os
want = os.path.basename('$1'.rstrip('/')).lower()
d = json.load(sys.stdin)
print(next((m['id'] for m in d['data'] if m['id'].lower().endswith(want)), ''))"
}
ID_A=$(resolve_id "$CHAT_A"); ID_B=$(resolve_id "$CHAT_B")
[ -n "$ID_A" ] || { echo -e "${RED}FAIL${NC} chat A not discovered ($CHAT_A)"; exit 1; }
[ -n "$ID_B" ] || { echo -e "${RED}FAIL${NC} chat B not discovered ($CHAT_B)"; exit 1; }
ID_TTS=""; ID_IMG=""; ID_A2=""
[ -n "$TTS" ] && ID_TTS=$(resolve_id "$TTS")
[ -n "$IMAGE" ] && ID_IMG=$(resolve_id "$IMAGE")
[ -n "${STRESS_CHAT_A2:-}" ] && [ -d "${STRESS_CHAT_A2}" ] && ID_A2=$(resolve_id "$STRESS_CHAT_A2")
echo "   chat A: $ID_A"
echo "   chat B: $ID_B"
[ -n "$ID_TTS" ] && echo "   tts:    $ID_TTS"
[ -n "$ID_IMG" ] && echo "   image:  $ID_IMG"
[ -n "$ID_A2" ] && echo "   chat A2 (even waves): $ID_A2"

mem_sample() {  # -> "active_mb cache_mb"
    curl -s --max-time 10 "$BASE/props" | python3 -c "
import sys, json
m = json.load(sys.stdin).get('memory', {})
print(m.get('active_bytes', 0) // (1 << 20), m.get('cache_bytes', 0) // (1 << 20))" 2>/dev/null || echo "0 0"
}

# ---------------------------------------------------------------- fire ------
TTS_N=0; IMG_N=0
[ -n "$ID_TTS" ] && TTS_N=$N_TTS
[ -n "$ID_IMG" ] && IMG_N=$N_IMG
SOAK_WAVES=${SOAK_WAVES:-1}
FAILED=0
MEM_WAVE2=""
for wave in $(seq 1 "$SOAK_WAVES"); do
    WAVE_A="$ID_A"
    [ -n "$ID_A2" ] && [ $((wave % 2)) -eq 0 ] && WAVE_A="$ID_A2"
    echo "== wave $wave/$SOAK_WAVES: $N_AGENTS agent($WAVE_A) + $N_PLAIN plain + $TTS_N tts + $IMG_N img + $N_ABORT abort + $N_CHURN churn =="
    PIDS=(); NAMES=()
    spawn() {  # kind label model rounds
        python3 "$TMP/worker.py" "$1" "$2" "$BASE" "$3" "$MAX_TOKENS" "$4" \
            > "$TMP/results/$2.out" 2>&1 &
        PIDS+=($!); NAMES+=("$2")
    }
    # NB: BSD seq counts DOWN when first > last (`seq 1 0` -> "1 0"), so every
    # count is gated before its loop.
    [ "$N_AGENTS" -gt 0 ] && for i in $(seq 1 "$N_AGENTS"); do spawn agent "w$wave-agent$i" "$WAVE_A" "$ROUNDS"; done
    [ "$N_PLAIN" -gt 0 ] && for i in $(seq 1 "$N_PLAIN"); do spawn plain "w$wave-plain$i" "$ID_B" "$ROUNDS"; done
    [ -n "$ID_TTS" ] && spawn tts "w$wave-tts" "$ID_TTS" "$N_TTS"
    [ -n "$ID_IMG" ] && spawn img "w$wave-img" "$ID_IMG" "$N_IMG"
    [ "$N_ABORT" -gt 0 ] && for i in $(seq 1 "$N_ABORT"); do spawn abort "w$wave-abort$i" "$WAVE_A" "$ROUNDS"; done
    [ "$N_CHURN" -gt 0 ] && for i in $(seq 1 "$N_CHURN"); do spawn churn "w$wave-churn$i" "$ID_B" "$ROUNDS"; done

    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then :; else FAILED=$((FAILED + 1)); fi
        cat "$TMP/results/${NAMES[$i]}.out"
    done

    if ! kill -0 "$SRV" 2>/dev/null; then
        echo -e "${RED}FAIL${NC} SERVER DIED in wave $wave. Last log lines:"
        tail -40 "$LOG" 2>/dev/null; SRV=""
        exit 1
    fi
    MEM=$(mem_sample)
    echo "   [mem] wave $wave: active=$(echo "$MEM" | cut -d' ' -f1) MB cache=$(echo "$MEM" | cut -d' ' -f2) MB"
    [ "$wave" -eq 2 ] && MEM_WAVE2=$(echo "$MEM" | cut -d' ' -f1)
done

# Soak memory bound: after warm-up (wave 2: every model resident, caches
# primed) active_bytes must not keep climbing — the leak class, not variance.
if [ "$SOAK_WAVES" -ge 3 ] && [ -n "$MEM_WAVE2" ] && [ "$MEM_WAVE2" -gt 0 ]; then
    FINAL_ACTIVE=$(mem_sample | cut -d' ' -f1)
    LIMIT=$((MEM_WAVE2 * 3 / 2))
    if [ "$FINAL_ACTIVE" -gt "$LIMIT" ]; then
        echo -e "${RED}FAIL${NC} soak memory growth: wave2 active=${MEM_WAVE2} MB -> final ${FINAL_ACTIVE} MB (> 1.5x)"
        exit 1
    fi
    echo "   [mem] soak bound OK: wave2=${MEM_WAVE2} MB, final=${FINAL_ACTIVE} MB (limit ${LIMIT} MB)"
fi

# ---------------------------------------------------------------- verify ----
echo "== post-stress verification =="
if ! kill -0 "$SRV" 2>/dev/null; then
    echo -e "${RED}FAIL${NC} SERVER DIED during stress (the crash class). Last log lines:"
    tail -40 "$LOG" 2>/dev/null; tail -20 "$TMP/stdout.log"
    SRV=""  # nothing to kill in cleanup
    exit 1
fi
curl -sf "$BASE/health" >/dev/null || { echo -e "${RED}FAIL${NC} /health dead after stress"; exit 1; }

FINAL=$(curl -s --max-time 300 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ID_A\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":16}")
echo "$FINAL" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['choices'][0]['message']['content'].strip(), 'empty post-stress reply'
print('   post-stress round-trip OK:', repr(d['choices'][0]['message']['content'][:40]))" \
    || { echo -e "${RED}FAIL${NC} post-stress chat broken: $(echo "$FINAL" | head -c 200)"; exit 1; }

if grep -nE "panic|Segmentation|Illegal instruction|std.terminate|libc\+\+abi" "$LOG" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC} crash signature in server log:"
    grep -nE "panic|Segmentation|Illegal instruction|std.terminate|libc\+\+abi" "$LOG" | head -5
    exit 1
fi

# Engagement: a leg that silently no-oped would PASS on transport alone — the
# server's own log must show the work actually ran (bench-rule discipline).
engaged() {  # pattern expected_count leg
    local got; got=$(grep -cE "$1" "$LOG" 2>/dev/null || true)
    if [ "${got:-0}" -lt "$2" ]; then
        echo -e "${RED}FAIL${NC} engagement: $3 — expected >= $2 '$1' lines in server log, got ${got:-0}"
        exit 1
    fi
}
engaged "model id=$ID_A ready" 1 "chat A load"
engaged "model id=$ID_B ready" 1 "chat B load"
[ -n "$ID_IMG" ] && engaged '\[image\] -> [0-9]+ PNG bytes' $((N_IMG * SOAK_WAVES)) "image generations"
[ -n "$ID_TTS" ] && engaged ' -> [0-9]+ WAV bytes' $((N_TTS * SOAK_WAVES)) "TTS generations"

if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}FAIL${NC} $FAILED worker(s) failed (server survived)"
    exit 1
fi
echo -e "${GREEN}PASS${NC} all workers clean, server alive and functional"
