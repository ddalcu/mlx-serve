#!/bin/bash
# FLUX.2 completeness contract (issue #362's fix family). The diffusers layout
# keeps every weight in component dirs, and the only MLX build of klein 9B
# (`mlx-community/flux2-klein-9b-4bit`) ships NO root config.json — the shape is
# proven by the DiT index's own tensor names. A copy starved of `vae/` must not
# read as present, so the next pull re-enters the manifest instead of fast-
# pathing into a `FileNotFound` dir. Cells need no network: the marker prints
# from LOCAL state before any network I/O.
set -u

BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"
    exit 0
fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
PASS=0
FAIL=0
check() {
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; PASS=$((PASS + 1)); else echo "FAIL: $1"; FAIL=$((FAIL + 1)); fi
}

# A syntactically valid safetensors file with one dummy tensor. `dit` carries
# the FLUX.2 shared-modulation tensor (the shape's identity marker); anything
# else carries a plain name (a real VAE or text-encoder shard is not a DiT).
fake_st() {
    python3 - "$1" "${2:-plain}" <<'PY'
import json, struct, sys
name = ("double_stream_modulation_img.linear_1x1.weight" if sys.argv[2] == "dit"
        else "conv_in.weight")
hdr = json.dumps({name: {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
data = bytearray(hdr)
while len(data) % 8: data += b' '
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<Q', len(hdr)) + bytes(data) + b'\x00' * 8)
PY
}

# The pull layout for the configless 9B build, starved of `vae/` (what the old
# selection produced: the DiT and text encoder landed, the VAE's dir never did).
MODEL="$SCRATCH/.mlx-serve/models/mlx-community/flux2-klein-9b-4bit"
mkdir -p "$MODEL/transformer" "$MODEL/text_encoder" "$MODEL/tokenizer"
printf '{"weight_map":{"double_stream_modulation_img.linear_1x1.weight":"0.safetensors"}}\n' \
    > "$MODEL/transformer/model.safetensors.index.json"
fake_st "$MODEL/transformer/0.safetensors" dit
printf '{"weight_map":{"decoder.conv_in.weight":"0.safetensors"}}\n' \
    > "$MODEL/text_encoder/model.safetensors.index.json"
fake_st "$MODEL/text_encoder/0.safetensors"
printf '{}\n' > "$MODEL/tokenizer/tokenizer.json"

# ── A. `list` sees it: discovery proves the shape from the DiT index alone
# (no root config.json) — the MageFlow class all over again.
if HOME="$SCRATCH" "$BIN" list 2>/dev/null | grep -q "flux2-klein-9b-4bit"; then G=0; else G=1; fi
check "list sees a configless FLUX.2 repo (DiT index proves the shape)" $G

# ── B. The starved copy is NOT present: `pull` re-enters the manifest fetch
# ("pulling manifest for …"), never the "model at …" fast path. The loader's
# fast path refuses media shapes by design, so probe through the puller like
# tests/test_partial_download.sh — marker first, kill right after it appears.
OUT="$SCRATCH/pull.txt"
HOME="$SCRATCH" "$BIN" pull mlx-community/flux2-klein-9b-4bit > "$OUT" 2>&1 &
PID=$!
SEEN=1
for _ in $(seq 1 40); do
    if grep -q "pulling manifest for mlx-community/flux2-klein-9b-4bit" "$OUT"; then SEEN=0; break; fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
if grep -q "pulling manifest" "$OUT"; then SEEN=0; fi
if grep -q "model at" "$OUT"; then FAST=1; else FAST=0; fi
check "starved FLUX.2 copy re-enters the pull, not the already-present fast path" $SEEN
check "starved FLUX.2 copy never takes the already-present fast path" $FAST

# ── C. The VAE lands: the same `pull` fast-paths ("model at …" with no
# manifest fetch). Completeness is what flipped, not the shape.
mkdir -p "$MODEL/vae"
printf '{"weight_map":{"vae_decoder.conv_in.weight":"0.safetensors"}}\n' \
    > "$MODEL/vae/model.safetensors.index.json"
fake_st "$MODEL/vae/0.safetensors"
OUT2="$SCRATCH/pull2.txt"
HOME="$SCRATCH" "$BIN" pull mlx-community/flux2-klein-9b-4bit > "$OUT2" 2>&1
if grep -q "model at" "$OUT2" && ! grep -q "pulling manifest" "$OUT2"; then AT=0; else AT=1; fi
check "complete FLUX.2 copy is present (fast path, no re-pull)" $AT

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
