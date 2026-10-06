#!/bin/bash
# test_tokenizer_hf_parity.sh — /tokenize equals HF `tokenizers` on code, per tokenizer family.
#
# Bar: zero differing tokens over this repo's Zig, Swift and JS sources (agent traffic). A
# pre-token grammar or BPE option we read wrong (the plain Llama-3 regex served with Muse's cased
# grammar; `ignore_merges`) puts every agent prompt off-distribution while nothing errors.
# Not covered: non-Latin scripts and Unicode numerics (`²`, Devanagari marks), where our letter /
# mark / digit tables are approximate ranges, and NFC normalizers.
#
#   TOKENIZERS_PYTHON=<python with `tokenizers`> ./tests/test_tokenizer_hf_parity.sh [pack...] [--port N]
#
# Default packs: one per pre-tokenizer family on this box (SKIPs the missing). SKIPs without a
# python that imports `tokenizers`.
set -uo pipefail
cd "$(dirname "$0")/.."
source tests/_lib_models.sh

PY="${TOKENIZERS_PYTHON:-python3}"
PORT=11368
PACKS=()
while [[ $# -gt 0 ]]; do
    case "$1" in --port) PORT="$2"; shift 2 ;; *) PACKS+=("$1"); shift ;; esac
done
"$PY" -c "import tokenizers" 2>/dev/null || { echo "SKIP: $PY cannot import tokenizers (set TOKENIZERS_PYTHON)"; exit 0; }
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -x "$BIN" ]] || { echo "FAIL: $BIN missing"; exit 1; }
if [[ ${#PACKS[@]} -eq 0 ]]; then
    for rel in TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP mlx-community/Llama-3.2-3B-Instruct-4bit \
               LiquidAI/LFM2.5-2.6B-MLX-mxfp4 ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit \
               mlx-community/Qwen3.5-0.8B-MLX-4bit mlx-community/gemma-4-e4b-it-8bit; do
        p=$(find_model "$rel") && PACKS+=("$p") || echo "SKIP: $rel not on this box"
    done
fi
WORK=$(mktemp -d)
trap 'kill $SRV 2>/dev/null; rm -rf "$WORK"' EXIT
SRV=""
fail=0
for pack in "${PACKS[@]}"; do
    HOME="$WORK" "$BIN" --model "$pack" --serve --host 127.0.0.1 --port "$PORT" > "$WORK/server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && break; sleep 1; done
    "$PY" - "$pack" "$PORT" src/server.zig src/chat.zig app/Sources/MLXServe/AppState.swift src/html/app.js <<'PY' || fail=1
import difflib, json, sys, urllib.request
from tokenizers import Tokenizer
pack, port, files = sys.argv[1], sys.argv[2], sys.argv[3:]
hf = Tokenizer.from_file(pack + "/tokenizer.json")
diffs, total = [], 0
for f in files:
    text = open(f, errors="replace").read()[:40000]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", json.dumps({"content": text, "add_special": False}).encode(),
                                 {"Content-Type": "application/json"})
    ours = json.load(urllib.request.urlopen(req, timeout=120))["tokens"]
    want = hf.encode(text, add_special_tokens=False).ids
    total += len(want)
    for op in difflib.SequenceMatcher(a=want, b=ours, autojunk=False).get_opcodes():
        if op[0] != "equal":
            diffs.append((hf.decode(want[op[1]:op[2]]), [hf.decode([t]) for t in ours[op[3]:op[4]]]))
name = pack.rstrip("/").split("/")[-1]
print(f"  {'PASS' if not diffs else 'FAIL'} {name}: {len(diffs)} differing spans over {total} tokens")
for want, got in diffs[:8]:
    print(f"       hf {want!r} -> ours {got}")
sys.exit(1 if diffs else 0)
PY
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
done
[[ $fail -eq 0 ]] && echo "tokenizer HF parity: PASS" || echo "tokenizer HF parity: FAIL"
exit $fail
