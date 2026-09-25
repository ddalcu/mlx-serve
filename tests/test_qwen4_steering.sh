#!/bin/bash
# Directional steering on the native qwen4_exp trunk:
# launch flags, the three per-model settings states, per-request `steering`,
# `POST /v1/steering` + the `steer` CLI, the MTP-invariance bar, the accepted
# stale-KV behaviour, the capture-to-bank round trip, and refusals by NAME.
# Numerical arms boot with the prefix cache OFF (a restored prefix carries the
# steering it was computed under, by design — arm [6] pins that).
#
# Runs under a private HOME (registry dir + model-settings.json sandboxed).
# NEEDS the real pack (70 GB, ten boots): skips without it.
#
# Usage: STEERING_MODEL=<dir> ./tests/test_qwen4_steering.sh [port]
set -u
MODEL="${STEERING_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}"
PORT="${1:-11416}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no pack at $MODEL"; exit 0; }
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
U="http://127.0.0.1:$PORT"
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }

FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.mlx-serve/steering" "$FAKE_HOME/logs"
SETTINGS="$FAKE_HOME/.mlx-serve/model-settings.json"
SPID=""
cleanup() { [ -n "$SPID" ] && { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }; rm -rf "$FAKE_HOME"; }
trap cleanup EXIT

# Synthetic banks at the pack's own geometry: a random unit bank (steers
# visibly) and a wrong-size file (refused by name).
python3 - "$MODEL" "$FAKE_HOME" <<'EOF'
import json, os, random, struct, sys, math
model, home = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(model, "config.json")))
tc = cfg.get("text_config", cfg)
L, H = tc["num_hidden_layers"], tc["hidden_size"]
random.seed(7)
rows = []
for _ in range(L):
    v = [random.random() - 0.5 for _ in range(H)]
    n = math.sqrt(sum(x * x for x in v))
    rows += [x / n for x in v]
d = os.path.join(home, ".mlx-serve", "steering")
open(os.path.join(d, "rnd.f32"), "wb").write(struct.pack("<%df" % len(rows), *rows))
open(os.path.join(home, "short.f32"), "wb").write(struct.pack("<%df" % (len(rows) - 1), *rows[:-1]))
open(os.path.join(home, "geometry"), "w").write(f"{L} {H}")
EOF
read -r N_LAYERS HIDDEN < "$FAKE_HOME/geometry"
WANT_BYTES=$((N_LAYERS * HIDDEN * 4))

boot() { # boot <log> [flags...]  (env: BOOT_ENV extra "K=V" pairs)
    local log="$1"; shift
    env HOME="$FAKE_HOME" ${BOOT_ENV:-} "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --log-level info --log-file off "$@" >"$log" 2>&1 &
    SPID=$!
    for _ in $(seq 1 900); do
        curl -sf -m 10 "$U/health" >/dev/null 2>&1 && grep -q "Model ready" "$log" && return 0
        kill -0 "$SPID" 2>/dev/null || { echo "server died:"; tail -5 "$log"; return 1; }
        sleep 1
    done
    echo "server never became healthy"; return 1
}
stop() { [ -n "$SPID" ] && { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""; }; sleep 3; }

# chat <extra-json-fields> -> prints "<summed logprob of the answer>|<text>". The sum over 24
# tokens is the meter: the first token of a greedy answer is near-certain, so its own
# logprob cannot show a steering shift. S is the scale the "moves" arms use: a random
# unit direction has a ~1/sqrt(hidden) component, so a mild scale is invisible.
PROMPT='Explain why databases use indexes.'
S=-8
cat > "$FAKE_HOME/sum.py" <<'PY'
import json, sys
j = json.load(sys.stdin)
if "choices" not in j:
    # Errors print HTTPERROR so a 'moved' check can't pass on one.
    print("HTTPERROR|" + str(j.get("error", {}).get("message", j))[:120])
    raise SystemExit(0)
c = j["choices"][0]
toks = (c.get("logprobs") or {}).get("content") or []
head = f"{sum(t['logprob'] for t in toks):.4f}" if toks else "nologprobs"
print(head + "|" + c["message"]["content"])
PY
chat() {
    curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":24,\"temperature\":0,\"enable_thinking\":false,\"logprobs\":true,\"top_logprobs\":1$1}" \
    | python3 "$FAKE_HOME/sum.py"
}
# status <extra-json-fields> -> HTTP status of a chat request
status() {
    curl -s -m 300 -o /dev/null -w '%{http_code}' "$U/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4$1}"
}
errmsg() {
    curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4$1}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("error",{}).get("message",""))'
}

echo "[1] unsteered boot: no [steering] line, GET reports the registry, per-request steering moves logits"
LOG1="$FAKE_HOME/logs/1.log"
boot "$LOG1" --prefix-cache-entries 0 || exit 1
check "no [steering] lines" "$(grep -c '\[steering\]' "$LOG1")" "0"
G=$(curl -s -m 300 "$U/v1/steering")
check "GET supported" "$(echo "$G" | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["supported"], j["active"], j["capture"], [r["name"] for r in j["registry"]])')" "True None False ['rnd']"
check "the mlx-serve alias names the default model" "$(curl -s -o /dev/null -w '%{http_code}' -m 30 "$U/v1/steering?model=mlx-serve")" "200"
BASE=$(chat "")
STEERED=$(chat ",\"steering\":{\"name\":\"rnd\",\"ffn\":$S}")
check "unsteered answer has a logprob sum (the meter itself)" "$(echo "$BASE" | grep -cE '^-?[0-9]+\.[0-9]{4}\|')" "1"
check "per-request steering changes the answer's logprob sum" "$([ "${BASE%%|*}" != "${STEERED%%|*}" ] && echo moved || echo same)" "moved"
OPTOUT=$(chat ',"steering":{"ffn":0,"attn":0}')
check "explicit opt-out == unsteered" "${OPTOUT%%|*}" "${BASE%%|*}"
check "bad name is a 400" "$(status ',"steering":{"name":"no such"}')" "400"
check "bad name names the rule" "$(errmsg ',"steering":{"name":"no such"}' | grep -c 'name must be')" "1"
check "wrong-size file is a 400 by name" "$(errmsg ",\"steering\":{\"file\":\"$FAKE_HOME/short.f32\"}" | grep -c 'file size')" "1"
check "capture without the env is a 400 by name" "$(errmsg ',"steering":{"ffn":0,"attn":0},"steering_capture":"x1"' | grep -c 'MLX_SERVE_STEERING_DUMP_DIR')" "1"
check "streaming request with a bad name is a 400, not a 200 stream" "$(status ',"stream":true,"steering":{"name":"no such"}')" "400"
echo "[2] POST /v1/steering flips the default for bare requests, persists, and the CLI turns it off"
P=$(curl -s -m 300 "$U/v1/steering" -H 'content-type: application/json' -d "{\"name\":\"rnd\",\"ffn\":$S}")
check "POST active" "$(echo "$P" | python3 -c 'import json,sys; a=json.load(sys.stdin)["active"]; print(a["name"], a["ffn"], a["attn"])')" "rnd $S 0"
check "active line logged" "$(grep -c '\[steering\] active: rnd.f32' "$LOG1")" "1"
BARE=$(chat "")
check "bare request follows the active default" "${BARE%%|*}" "${STEERED%%|*}"
check "settings file carries it" "$(python3 -c "import json; d=json.load(open('$SETTINGS')); print([v.get('steering',{}).get('name') for k,v in d.items()])")" "['rnd']"
check "props settings name the file" "$(curl -s -m 300 "$U/props" | python3 -c 'import json,sys; s=json.load(sys.stdin)["settings"]["steering"]; print(s["file"].endswith("rnd.f32"), s["ffn"])')" "True $S"
CLI=$(HOME="$FAKE_HOME" "$BIN" steer --port "$PORT" off)
check "CLI off" "$(echo "$CLI" | python3 -c 'import json,sys; print(json.load(sys.stdin)["active"])')" "None"
AFTER=$(chat "")
check "bare request is unsteered again" "${AFTER%%|*}" "${BASE%%|*}"
check "settings file carries off (null)" "$(python3 -c "import json; d=json.load(open('$SETTINGS')); print([v.get('steering','absent') for k,v in d.items()])")" "[None]"
check "reset beside a bank is a 400 by name" "$(curl -s -m 30 "$U/v1/steering" -H 'content-type: application/json' -d '{"reset":true,"name":"rnd"}' | grep -c 'reset takes no')" "1"
# A cold model still answers GET from its arch, and a POST needs it resident.
curl -s -m 300 "$U/v1/unload-model" -X POST -H 'content-type: application/json' -d '{"model":"mlx-serve"}' >/dev/null
check "cold GET: supported, not steerable" "$(curl -s -m 30 "$U/v1/steering" | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["supported"], j["steerable"])')" "True False"
check "cold POST is a 409" "$(curl -s -o /dev/null -w '%{http_code}' -m 30 "$U/v1/steering" -H 'content-type: application/json' -d '{"name":"rnd","ffn":1}')" "409"
stop
echo "[3] MTP-invariance: a steered greedy answer is the same with and without MTP"
# Its OWN boot with --mtp: qwen4 keeps the in-checkpoint head opt-in (MoE default-off),
# so without the flag `enable_mtp:true` arms nothing and both arms decode serial.
LOG3="$FAKE_HOME/logs/3.log"
BOOT_ENV="MLX_SERVE_MTP_FORCE_DEPTH=2" boot "$LOG3" --mtp --prefix-cache-entries 0 || exit 1
# NO logprobs here: `requestSpecModes` sets spec_ok = logprobs_n == 0, so the meter that
# every other arm uses would disable MTP in BOTH arms and compare serial with serial.
# Text only, and the engagement count below is what proves MTP actually ran.
mtpchat() {
    curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":24,\"temperature\":0,\"enable_thinking\":false$1}" \
    | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["choices"][0]["message"]["content"] if "choices" in j else "HTTPERROR")'
}
# `grep -c` exits 1 on a zero count; counts go into plain variables before comparing.
MTP_BEFORE=$(grep -c 'spec-stats\] mode=mtp' "$LOG3" 2>/dev/null || true)
PLAIN=$(mtpchat ',"enable_mtp":true')
MTP=$(mtpchat ",\"steering\":{\"name\":\"rnd\",\"ffn\":$S},\"enable_mtp\":true")
MTP_AFTER=$(grep -c 'spec-stats\] mode=mtp' "$LOG3" 2>/dev/null || true)
if [ "$MTP_AFTER" -gt "$MTP_BEFORE" ]; then MTP_ENGAGED=engaged; else MTP_ENGAGED=never; fi
check "the MTP arm actually engaged MTP" "$MTP_ENGAGED" "engaged"
# The steer must be visible, or equality below would also hold with verify rows unsteered.
check "steering moves the MTP answer" "$([ "$MTP" != "$PLAIN" ] && echo moved || echo same)" "moved"
# Verify rows run other kernels than serial decode, so greedy may flip at a near-tie
# (auto depth is not even self-reproducible; FORCE_DEPTH pins it). A divergence is
# acquitted only where serial's top-2 gap is small and MTP took serial's runner-up. The
# bar is looser than test_mtp_equivalence's 0.15: a strong steer flattens the top of the
# distribution.
SERIAL_LP=$(curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":24,\"temperature\":0,\"enable_thinking\":false,\"logprobs\":true,\"top_logprobs\":2,\"steering\":{\"name\":\"rnd\",\"ffn\":$S}}")
VERDICT=$(MTP="$MTP" python3 -c '
import json, os, sys
toks = json.load(sys.stdin)["choices"][0]["logprobs"]["content"]
mtp, at = os.environ["MTP"], 0
serial = "".join(t["token"] for t in toks)
lead = serial[:len(serial) - len(serial.lstrip())]  # content is lead-trimmed, tokens are not
if not mtp.startswith(lead): mtp = lead + mtp
if mtp and mtp == serial.rstrip("\n"):  # $(...) strips the trailing newlines
    print("identical"); sys.exit()
for t in toks:
    if not mtp.startswith(t["token"], at):
        top = t["top_logprobs"]
        near = len(top) > 1 and top[0]["logprob"] - top[1]["logprob"] <= 0.3 and mtp.startswith(top[1]["token"], at)
        print("near-tie" if near else "diverged at %d: %r" % (at, mtp[at:at + 20])); break
    at += len(t["token"])
else: print("diverged: lengths differ")' <<< "$SERIAL_LP")
check "steered mtp == no-mtp up to a serial near-tie" "$([ "$VERDICT" = identical ] || [ "$VERDICT" = near-tie ] && echo ok || echo "$VERDICT")" "ok"
stop

echo "[4] launch flags arm the default; a persisted off beats them; a persisted config arms without flags"
rm -f "$SETTINGS"
LOG4="$FAKE_HOME/logs/4.log"
boot "$LOG4" --prefix-cache-entries 0 --dir-steering-file rnd --dir-steering-ffn "$S" || exit 1
check "armed line from flags" "$(grep -c '\[steering\] armed: rnd.f32 layers='"$N_LAYERS"' hidden='"$HIDDEN"' ffn='"$S"' attn=0 (source=flags)' "$LOG4")" "1"
FLAGGED=$(chat "")
check "flag-armed bare request == the per-request steered answer" "${FLAGGED%%|*}" "${STEERED%%|*}"
OPT=$(chat ',"steering":{"ffn":0,"attn":0}')
check "opt-out under flags == unsteered" "${OPT%%|*}" "${BASE%%|*}"
# A flag bank deleted under a running server: reset still clears, and serves unsteered.
cp "$FAKE_HOME/.mlx-serve/steering/rnd.f32" "$FAKE_HOME/rnd.keep"
rm "$FAKE_HOME/.mlx-serve/steering/rnd.f32"
check "reset with the flag bank gone is a 200" "$(curl -s -o /dev/null -w '%{http_code}' -m 60 "$U/v1/steering" -H 'content-type: application/json' -d '{"reset":true}')" "200"
check "and it is logged, not silent" "$(grep -c 'reset: launch-flag bank rnd unusable' "$LOG4")" "1"
mv "$FAKE_HOME/rnd.keep" "$FAKE_HOME/.mlx-serve/steering/rnd.f32"
stop
echo "{ \"$MODEL\": { \"steering\": null } }" > "$SETTINGS"
LOG4B="$FAKE_HOME/logs/4b.log"
boot "$LOG4B" --prefix-cache-entries 0 --dir-steering-file rnd --dir-steering-ffn "$S" || exit 1
check "persisted off + flags: zero [steering] lines" "$(grep -c '\[steering\]' "$LOG4B")" "0"
check "persisted off + flags: GET active null" "$(curl -s -m 300 "$U/v1/steering" | python3 -c 'import json,sys; print(json.load(sys.stdin)["active"])')" "None"
stop
echo "{ \"$MODEL\": { \"steering\": {\"name\": \"rnd\", \"ffn\": $S} } }" > "$SETTINGS"
LOG4C="$FAKE_HOME/logs/4c.log"
boot "$LOG4C" --prefix-cache-entries 0 || exit 1
check "armed line from settings" "$(grep -c 'ffn='"$S"' attn=0 (source=settings)' "$LOG4C")" "1"
SET=$(chat "")
check "settings-armed bare request == the per-request steered answer" "${SET%%|*}" "${STEERED%%|*}"
stop
rm -f "$SETTINGS"

echo "[5] capture boot: dumped rows build a $WANT_BYTES-byte bank from two prompt pairs"
CAP="$FAKE_HOME/cap"; mkdir -p "$CAP"
LOG5="$FAKE_HOME/logs/5.log"
BOOT_ENV="MLX_SERVE_STEERING_DUMP_DIR=$CAP" boot "$LOG5" --max-concurrent 1 --prefix-cache-entries 0 || exit 1
check "GET capture armed" "$(curl -s -m 300 "$U/v1/steering" | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["capture"], j["capture_dir"])')" "True $CAP"
check "capture while steered is a 400 by name" "$(errmsg ',"steering":{"name":"rnd","ffn":1},"steering_capture":"x2"' | grep -c 'both arms off')" "1"
check "capture on a stream is a 400 by name" "$(errmsg ',"stream":true,"steering":{"ffn":0,"attn":0},"steering_capture":"x3"' | grep -c 'stream:false')" "1"
curl -s -m 300 -o /dev/null "$U/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0,"enable_thinking":false,"steering":{"ffn":0,"attn":0},"steering_capture":"x5"}'
check "a capture dumps all layers" "$(ls "$CAP/x5"/ffn_out-*_pos0.bin 2>/dev/null | wc -l | tr -d ' ')" "$N_LAYERS"
check "its manifest names the capture" "$(python3 -c "import json; print(json.load(open('$CAP/x5/done.json'))['capture_id'])" 2>/dev/null)" "x5"
# The SHIPPED builder on the SHIPPED example prompts, so the documented recipe is what
# runs here. Two pairs rather than all 16: each capture is a cold prefill, and the bar is
# the round trip, not the bank's quality.
head -2 "$ROOT/tests/fixtures/steering/succinct.txt" > "$FAKE_HOME/good.txt"
head -2 "$ROOT/tests/fixtures/steering/verbose.txt" > "$FAKE_HOME/bad.txt"
python3 "$ROOT/tests/build_steering_bank.py" --base "$U" --dump-dir "$CAP" \
    --show "$FAKE_HOME/good.txt" --control "$FAKE_HOME/bad.txt" \
    --out "$FAKE_HOME/.mlx-serve/steering/verbosity" > "$FAKE_HOME/logs/builder.log" 2>&1
check "builder exit" "$?" "0"
check "bank bytes" "$(stat -f %z "$FAKE_HOME/.mlx-serve/steering/verbosity.f32" 2>/dev/null)" "$WANT_BYTES"
check "bank metadata" "$(python3 -c "import json; j=json.load(open('$FAKE_HOME/.mlx-serve/steering/verbosity.json')); print(j['format'], j['shape'], j['show'], j['control'], j['component'], j['orthogonalized'])")" "directional-steering-v1 [$N_LAYERS, $HIDDEN] 2 2 ffn_out True"
# A --dump-dir that is not the server's is refused by name, not read as empty rows.
python3 "$ROOT/tests/build_steering_bank.py" --base "$U" --dump-dir "$FAKE_HOME/empty-caps" \
    --run r2 --show "$FAKE_HOME/good.txt" --control "$FAKE_HOME/bad.txt" \
    --out "$FAKE_HOME/nope" > "$FAKE_HOME/logs/builder_bad.log" 2>&1
check "builder refuses a wrong dump dir by name" "$(grep -c 'directory the server was started with' "$FAKE_HOME/logs/builder_bad.log")" "1"
check "no builder capture was skipped" "$(grep -c 'capture skipped' "$LOG5")" "0"
# Same-boot baseline: a zero bank must read "same" here.
BASE5=$(chat "")
V=$(chat ',"steering":{"name":"verbosity","ffn":-2}')
check "the built bank steers" "$([ "${V%%|*}" != "${BASE5%%|*}" ] && echo moved || echo same)" "moved"
stop

echo "[6] stale KV by design: a scale change mid-conversation reuses the cached prefix"
LOG6="$FAKE_HOME/logs/6.log"
boot "$LOG6" || exit 1
LONG=$(python3 -c 'print("Summarize this. " + "The quick brown fox jumps over the lazy dog. " * 60)')
CONV="{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"max_tokens\":8,\"temperature\":0,\"enable_thinking\":false"
curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$CONV,\"steering\":{\"name\":\"rnd\",\"ffn\":0.5}}" >/dev/null
CACHED=$(curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$CONV,\"steering\":{\"name\":\"rnd\",\"ffn\":-0.5}}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["prompt_tokens_details"]["cached_tokens"] > 0)')
check "second scale reuses the prefix (cached_tokens > 0)" "$CACHED" "True"
stop

echo "[6b] a steered request is unchanged by a concurrent plain one"
LOG6B="$FAKE_HOME/logs/6b.log"
boot "$LOG6B" --prefix-cache-entries 0 || exit 1
SOLO=$(chat ",\"steering\":{\"name\":\"rnd\",\"ffn\":$S}")
check "the steered arm moves at all" "$([ "${SOLO%%|*}" != "${BASE%%|*}" ] && echo moved || echo same)" "moved"
# A steered request must not be disturbed by a concurrent plain one. Logprob slots decode
# serial (BatchVerdict .logprobs), so this never shares a batch; batched steering is covered
# by the hermetic `qwen4 steering:` test.
# Detached and bounded: no `wait`, which blocks this shell indefinitely on macOS when the
# background child is reparented.
( curl -s -m 120 "$U/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly from one to forty in words.\"}],\"max_tokens\":220,\"temperature\":0,\"enable_thinking\":false}" >/dev/null 2>&1 & )
sleep 1
BESIDE=$(chat ",\"steering\":{\"name\":\"rnd\",\"ffn\":$S}")
check "a steered request is unchanged by a concurrent plain one" "$([ "${SOLO%%|*}" = "${BESIDE%%|*}" ] && echo same || echo changed)" "same"
check "the batched arms are real responses" "$(echo "$SOLO$BESIDE" | grep -c HTTPERROR)" "0"
stop

echo "[6c] the fold is byte-identical to the explicit seams on the REAL pack (BOTH arms)"
# The tiny-pack bar cannot see this: hidden 64 fails hcReadFused's %256 gate, so the fold
# never engages there and fold-vs-unfolded compares the standalone kernel with itself.
LOG6C="$FAKE_HOME/logs/6c.log"
boot "$LOG6C" --prefix-cache-entries 0 || exit 1
FOLDED=$(chat ",\"steering\":{\"name\":\"rnd\",\"ffn\":$S,\"attn\":$S}")
check "the fold engaged" "$(grep -c 'folded into the hyper-connection read' "$LOG6C")" "1"
stop
LOG6D="$FAKE_HOME/logs/6d.log"
BOOT_ENV="MLX_SERVE_STEER_FOLD=0" boot "$LOG6D" --prefix-cache-entries 0 || exit 1
UNFOLDED=$(chat ",\"steering\":{\"name\":\"rnd\",\"ffn\":$S,\"attn\":$S}")
check "the fold stood down under the kill switch" "$(grep -c 'folded into the hyper-connection read' "$LOG6D")" "0"
check "folded == explicit seams, byte for byte" "$([ "$FOLDED" = "$UNFOLDED" ] && echo same || echo differs)" "same"
stop

echo "[7] a bad launch file is a named load failure"
LOG7="$FAKE_HOME/logs/7.log"
HOME="$FAKE_HOME" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-file off --dir-steering-file "$FAKE_HOME/short.f32" >"$LOG7" 2>&1
check "boot with a wrong-size bank exits non-zero" "$([ $? -ne 0 ] && echo nonzero || echo zero)" "nonzero"
check "load failure names SteeringFileSize" "$(grep -q SteeringFileSize "$LOG7" && echo found)" "found"

echo "passed $pass failed $fail"
[ "$fail" = 0 ]
