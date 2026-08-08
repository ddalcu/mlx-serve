#!/usr/bin/env bash
# Multi-LoRA stacking across every backend that takes adapters, over the API.
#
# The claim under test is not "a LoRA attaches" (test_image_gen.sh already pins
# that for one file) but that SEVERAL attach at once and their deltas SUM.
# That is provable exactly, with no eyeballing and no reference image:
#
#   two IDENTICAL adapters at scale 1.0 must produce byte-for-byte the same
#   output as ONE of them at scale 2.0
#
# because d+d == 2*d exactly in IEEE floating point. A "last one wins" merge
# gives d, a dropped second file gives d, a doubled first file gives 2d from
# the wrong place — every one of those breaks the equality. The companion
# inequality (one adapter at 1.0 differs from 2.0) proves the delta is real,
# so the equality above cannot pass vacuously on a no-op adapter.
#
# Every generation runs at MINIMUM settings: the pixels are never looked at,
# only compared, so steps/size/frames are as low as each backend allows.
#
# That last sentence was also this suite's blind spot, and it cost a live bug:
# NOISE passes every comparison above. A real community LoRA rendered pure
# static (its alpha lived in the file metadata, so it ran 8x too strong) while
# all 56 checks here were green. `tests/lora_noise.py` is the companion that
# LOOKS — one number per render, bar 20 — applied to the generations that are
# supposed to be usable pictures.
#
# Usage: ./tests/test_multi_lora.sh [port]
set -uo pipefail
PORT="${1:-11466}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
MODELS="${MLX_SERVE_MODELS:-$HOME/.mlx-serve/models}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-lora.XXXXXX")"
LOG="$TMP/server.log"
SRV=""
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT
rc=0
declare -a ROWS

row() { ROWS+=("$1|$2|$3"); }        # model | check | verdict
ok()  { echo "  PASS: $2"; row "$1" "$2" "PASS"; }
bad() { echo "  FAIL: $2"; row "$1" "$2" "FAIL"; rc=1; }
skip(){ echo "  SKIP: $2"; row "$1" "$2" "SKIP"; }

# ── synthetic adapters ──
# `emit_lora <path> <module> <out> <in> <bvalue>`: a rank-2 adapter on ONE
# module. The dims are the REAL linear's, read once from each checkpoint's
# safetensors header and written down here rather than probed: for a packed
# weight, `cols*32 = in*bits` and the scales only give `in/group_size`, so
# 4-bit at group 64 and 8-bit at group 32 are indistinguishable from shapes
# alone — and only some of these packs declare `quantization` in config.json.
# To add a pack: read `<module>.weight` rows for `out`, and take `in` from the
# arch's hidden size (a wrong `in` fails loudly at the matmul, never silently).
# b=0 makes the delta exactly zero (transparency check); b>0 makes it
# real. Written as f32 safetensors, the shape the loader expects
# (A [r,in], B [out,r]).
emit_lora() {
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import sys, json, struct
path, module, out_dim, in_dim, bval = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
r = 2
a = struct.pack(f"<{r*in_dim}f", *([0.01] * (r * in_dim)))
b = struct.pack(f"<{out_dim*r}f", *([bval] * (out_dim * r)))
tensors = {
    f"{module}.lora_A.weight": ([r, in_dim], a),
    f"{module}.lora_B.weight": ([out_dim, r], b),
}
header, blob = {}, b""
for name, (shape, data) in tensors.items():
    header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
    blob += data
hj = json.dumps(header).encode()
open(path, "wb").write(struct.pack("<Q", len(hj)) + hj + blob)
PY
}

emit_foreign() {  # keys no backend knows → must 400
  python3 - "$1" <<'PY'
import sys, json, struct
z = struct.pack("<16f", *([0.0] * 16))
header, blob = {}, b""
for name, shape in [("unet.not_a_real_module.lora_A.weight", [2, 8]),
                    ("unet.not_a_real_module.lora_B.weight", [8, 2])]:
    header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(blob), len(blob) + len(z)]}
    blob += z
hj = json.dumps(header).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(hj)) + hj + blob)
PY
}

boot() {
  "$BIN" --serve --port "$PORT" --model-dir "$MODELS" >"$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server died"; tail -5 "$LOG"; return 1; }
    sleep 0.5
  done
  echo "FAIL: server never became healthy"; return 1
}
have() { [ -d "$MODELS/$1" ]; }

# One server serves every backend in turn, so without this each finished
# backend stays resident and the next one loads on top of it. Four image/video
# models later, MiniMax-H3's staged-residency preflight sees 39.01 GB free
# against a 38.97 GB peak and refuses the load — a 0.1% margin, i.e. a coin
# flip decided by whatever else the Mac is doing, reported as five failed LoRA
# checks. Nothing here needs two backends resident at once.
unload() { curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/unload-model" \
  -H 'Content-Type: application/json' -d "{\"model\":\"$1\"}"; }

# not_noise <label> <response.json> <what>: the render is a picture, not static.
# Exit 3 = numpy/PIL missing, which SKIPs loudly rather than passing silently.
not_noise() {
  local m rc
  m=$(python3 "$ROOT/tests/lora_noise.py" "$2" 2>&1); rc=$?
  case $rc in
    0) ok   "$1" "$3 is not noise ($m)" ;;
    3) skip "$1" "$3 noise check: $m" ;;
    *) bad  "$1" "$3 is NOISE ($m, bar 20)" ;;
  esac
}

# post <endpoint> <outfile> <json>  → echoes the http code
post() {
  curl -s --max-time 3000 -o "$2" -w '%{http_code}' -X POST \
    "http://127.0.0.1:$PORT/$1" -H 'Content-Type: application/json' -d "$3"
}

boot || exit 1

# ════════════════════════════════════════════════════════════════════════
# One backend's full matrix. The caller supplies the endpoint, the model id,
# the module to adapt, its dims, and the generation knobs (already minimal).
# ════════════════════════════════════════════════════════════════════════
run_matrix() { # <label> <endpoint> <model> <module> <out> <in> <extra-json> [level]
  local label="$1" ep="$2" model="$3" module="$4" out="$5" indim="$6" extra="$7"
  # "core" drops the checks whose claim is architecture-independent (they are
  # proven on the cheap backends) and keeps the ones that could differ here.
  # MiniMax-H3 re-stages text encoder + DiT for EVERY generation, so each
  # request costs a full load — the matrix is priced per request, not per run.
  local level="${8:-full}"
  local A="$TMP/$label-a.safetensors" B="$TMP/$label-b.safetensors"
  local Z="$TMP/$label-zero.safetensors" F="$TMP/$label-foreign.safetensors"
  emit_lora "$A" "$module" "$out" "$indim" 0.002
  emit_lora "$B" "$module" "$out" "$indim" 0.002   # byte-identical twin
  emit_lora "$Z" "$module" "$out" "$indim" 0.0
  emit_foreign "$F"

  echo "== $label"
  local base="{\"model\":\"$model\",\"seed\":7,$extra"

  # [1] baseline, no adapter
  local code
  code=$(post "$ep" "$TMP/$label.plain.json" "$base}")
  [ "$code" = "200" ] || { bad "$label" "baseline generates (http $code: $(head -c 160 "$TMP/$label.plain.json"))"; return; }
  ok "$label" "baseline generates"
  not_noise "$label" "$TMP/$label.plain.json" "baseline"

  if [ "$level" = full ]; then
    # [2] a zero-B adapter attaches and is numerically transparent — proves the
    # attach path does not disturb the base weights it wraps.
    code=$(post "$ep" "$TMP/$label.zero.json" "$base,\"lora_paths\":[\"$Z\"]}")
    if [ "$code" = "200" ] && cmp -s "$TMP/$label.plain.json" "$TMP/$label.zero.json"; then
      ok "$label" "zero-B adapter is byte-transparent"
    else
      bad "$label" "zero-B adapter changed the output or failed (http $code)"
    fi
  fi

  # [3] a real adapter at 1.0 changes the output — without this the equality
  # in [4] could pass on a delta that is silently zero.
  code=$(post "$ep" "$TMP/$label.one.json" "$base,\"lora_paths\":[\"$A\"],\"lora_scales\":[1.0]}")
  if [ "$code" = "200" ] && ! cmp -s "$TMP/$label.plain.json" "$TMP/$label.one.json"; then
    ok "$label" "one adapter changes the output"
    not_noise "$label" "$TMP/$label.one.json" "one adapter"
  else
    bad "$label" "one adapter did nothing (http $code)"
  fi

  # [4] scale is honoured: 2.0 differs from 1.0.
  code=$(post "$ep" "$TMP/$label.two.json" "$base,\"lora_paths\":[\"$A\"],\"lora_scales\":[2.0]}")
  if [ "$code" = "200" ] && ! cmp -s "$TMP/$label.one.json" "$TMP/$label.two.json"; then
    ok "$label" "lora_scales changes the output"
  else
    bad "$label" "scale 2.0 == scale 1.0 (http $code)"
  fi

  # [5] THE STACKING CLAIM: two identical adapters at 1.0 == one at 2.0,
  # byte for byte (d+d is exactly 2d).
  code=$(post "$ep" "$TMP/$label.stack.json" "$base,\"lora_paths\":[\"$A\",\"$B\"],\"lora_scales\":[1.0,1.0]}")
  if [ "$code" = "200" ] && cmp -s "$TMP/$label.two.json" "$TMP/$label.stack.json"; then
    ok "$label" "TWO adapters SUM (A@1+B@1 == A@2, byte-identical)"
    # Covers the scale-2.0 render above too — it is byte-identical to this one.
    not_noise "$label" "$TMP/$label.stack.json" "stacked pair"
  else
    bad "$label" "stacked pair != single at 2.0 — deltas are not summing (http $code)"
  fi

  if [ "$level" = full ]; then
    # [6] order is irrelevant, because a sum is commutative — a merge would not be.
    code=$(post "$ep" "$TMP/$label.rev.json" "$base,\"lora_paths\":[\"$B\",\"$A\"],\"lora_scales\":[1.0,1.0]}")
    if [ "$code" = "200" ] && cmp -s "$TMP/$label.stack.json" "$TMP/$label.rev.json"; then
      ok "$label" "stack order does not change the result"
    else
      bad "$label" "reversing the stack changed the output (http $code)"
    fi
  fi

  if [ "$level" = full ]; then
    # [7] mixed scales still sum: A@1 + B@1 must equal A@0.5 + B@1.5 (both 2d).
    code=$(post "$ep" "$TMP/$label.mixed.json" "$base,\"lora_paths\":[\"$A\",\"$B\"],\"lora_scales\":[0.5,1.5]}")
    if [ "$code" = "200" ] && cmp -s "$TMP/$label.stack.json" "$TMP/$label.mixed.json"; then
      ok "$label" "per-file scales fold into the sum (0.5+1.5 == 1+1)"
    else
      bad "$label" "mixed scales did not sum to the same delta (http $code)"
    fi
  fi

  if [ "$level" = full ]; then
    # [8] the legacy singular field still behaves exactly like a one-file array.
    code=$(post "$ep" "$TMP/$label.legacy.json" "$base,\"lora_path\":\"$A\",\"lora_scale\":2.0}")
    if [ "$code" = "200" ] && cmp -s "$TMP/$label.two.json" "$TMP/$label.legacy.json"; then
      ok "$label" "legacy lora_path/lora_scale unchanged"
    else
      bad "$label" "legacy single-adapter form drifted (http $code)"
    fi
  fi

  # [9] an adapter for another architecture is a named 400, never a silent
  # generation that ignores it.
  code=$(post "$ep" "$TMP/$label.foreign.json" "$base,\"lora_paths\":[\"$F\"]}")
  if [ "$code" = "400" ]; then ok "$label" "foreign adapter is a 400"
  else bad "$label" "foreign adapter returned $code (want 400)"; fi

  if [ "$level" = full ]; then
    # [10] over the cap → 400 (9 paths, max 8).
    local many="\"$A\",\"$A\",\"$A\",\"$A\",\"$A\",\"$A\",\"$A\",\"$A\",\"$A\""
    code=$(post "$ep" "$TMP/$label.many.json" "$base,\"lora_paths\":[$many]}")
    if [ "$code" = "400" ]; then ok "$label" "9 adapters is a 400 (cap 8)"
    else bad "$label" "over-cap stack returned $code (want 400)"; fi
  fi

  # [11] a relative path never reaches mlx (an MLX error kills the process).
  code=$(post "$ep" "$TMP/$label.rel.json" "$base,\"lora_paths\":[\"rel/nope.safetensors\"]}")
  if [ "$code" = "400" ]; then ok "$label" "relative path is a 400"
  else bad "$label" "relative path returned $code (want 400)"; fi

  unload "$model"
}

# ── IMAGE: FLUX.2 klein-4B ────────────────────────────────────────────────
if have "Runpod/FLUX.2-klein-4B-mflux-4bit"; then
  run_matrix "flux2-klein-4b" "v1/images/generations" "Runpod/FLUX.2-klein-4B-mflux-4bit" \
    "transformer.transformer_blocks.0.attn.to_q" 3072 3072 \
    "\"prompt\":\"a red apple\",\"size\":\"512x512\",\"steps\":2"
else skip "flux2-klein-4b" "not downloaded"; fi

# ── IMAGE: FLUX.2 klein-9b (same arch, wider: 32 heads x 128 = 4096) ──────
if have "mlx-community/flux2-klein-9b-4bit"; then
  run_matrix "flux2-klein-9b" "v1/images/generations" "mlx-community/flux2-klein-9b-4bit" \
    "transformer.transformer_blocks.0.attn.to_q" 4096 4096 \
    "\"prompt\":\"a red apple\",\"size\":\"512x512\",\"steps\":2"
else skip "flux2-klein-9b" "not downloaded"; fi

# ── IMAGE: Krea-2-Turbo ───────────────────────────────────────────────────
if have "ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8"; then
  run_matrix "krea2-turbo" "v1/images/generations" "ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8" \
    "blocks.0.attn.wq" 6144 6144 \
    "\"prompt\":\"a red apple\",\"size\":\"512x512\",\"steps\":2"
else skip "krea2-turbo" "not downloaded"; fi

# ── VIDEO: LTX-2.3 (8N+1 ladder → 9 frames is its floor) ──────────────────
if have "dgrauet/ltx-2.3-mlx-q4"; then
  run_matrix "ltx-2.3" "v1/video/generations" "dgrauet/ltx-2.3-mlx-q4" \
    "transformer.transformer_blocks.0.attn1.to_q" 4096 4096 \
    "\"prompt\":\"a red apple\",\"width\":256,\"height\":256,\"num_frames\":9,\"steps\":2"
else skip "ltx-2.3" "not downloaded"; fi

# ── VIDEO: MiniMax-H3 (17k+5 ladder → 5 frames; turbo floor is 4 steps) ───
H3=ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit
if have "$H3"; then
  run_matrix "minimax-h3" "v1/video/generations" "$H3" \
    "blocks.0.attn.qkv_proj" 21504 5376 \
    "\"prompt\":\"a red apple\",\"width\":256,\"height\":256,\"num_frames\":5,\"steps\":4" core

  # H3 only: the engine-owned Turbo adapter, and the thing no other backend
  # can be asked — does a user's adapter STACK with it?
  echo "== minimax-h3 turbo"
  TA="$TMP/minimax-h3-a.safetensors"
  H3B="{\"model\":\"$H3\",\"seed\":7,\"prompt\":\"a red apple\",\"width\":256,\"height\":256,\"num_frames\":5,\"steps\":4"

  if [ -f "$MODELS/$H3/turbo_lora.safetensors" ]; then
    code=$(post "v1/video/generations" "$TMP/h3.turbo.json" "$H3B,\"turbo\":true}")
    if [ "$code" = "200" ]; then
      ok "minimax-h3" "turbo generates"
      not_noise "minimax-h3" "$TMP/h3.turbo.json" "turbo"
    else bad "minimax-h3" "turbo failed (http $code)"; fi
    # It must attach to every module it names; a partial file is a broken
    # artifact, not a smaller speedup.
    if grep -q "lora 1/1: 259/259 modules" "$LOG"; then
      ok "minimax-h3" "turbo attaches all 259 modules"
    else
      bad "minimax-h3" "turbo attach count wrong: $(grep -o 'lora 1/1: [0-9]*/[0-9]*' "$LOG" | tail -1)"
    fi
    # Turbo must CHANGE the render (4 steps distilled vs 4 steps stock).
    code=$(post "v1/video/generations" "$TMP/h3.noturbo.json" "$H3B}")
    if [ "$code" = "200" ] && ! cmp -s "$TMP/h3.turbo.json" "$TMP/h3.noturbo.json"; then
      ok "minimax-h3" "turbo changes the output vs the same steps without it"
    else
      bad "minimax-h3" "turbo output identical to non-turbo (http $code)"
    fi
    # Turbo + a style adapter: BOTH files must attach (2/2 in the log) and the
    # result must differ from turbo alone.
    # Count BEFORE: the core matrix already sent a two-style-adapter request,
    # which logs "lora 2/2" too — a bare grep would match that one and pass
    # whether or not turbo stacked. Only a NEW line counts.
    before=$(grep -c "lora 2/2" "$LOG")
    code=$(post "v1/video/generations" "$TMP/h3.turbostack.json" "$H3B,\"turbo\":true,\"lora_paths\":[\"$TA\"]}")
    after=$(grep -c "lora 2/2" "$LOG")
    turbo_first=$(grep "lora 1/2" "$LOG" | tail -1 | grep -c "(turbo)")
    if [ "$code" = "200" ] && [ "$after" -gt "$before" ] && [ "$turbo_first" = "1" ] \
       && ! cmp -s "$TMP/h3.turbo.json" "$TMP/h3.turbostack.json"; then
      ok "minimax-h3" "a style adapter STACKS on top of turbo"
    else
      bad "minimax-h3" "turbo + style adapter did not stack (http $code)"
    fi
  else
    skip "minimax-h3" "turbo_lora.safetensors not in the pack"
  fi
else skip "minimax-h3" "not downloaded"; fi

# ── A backend that declares NO adapter support must refuse, not ignore ────
MF=ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit
if have "$MF"; then
  echo "== mage-flow (no adapter support)"
  emit_lora "$TMP/mf.safetensors" "transformer_blocks.0.attn.to_q" 3072 3072 0.002
  code=$(post "v1/images/generations" "$TMP/mf.json" \
    "{\"model\":\"$MF\",\"prompt\":\"a red apple\",\"size\":\"512x512\",\"steps\":4,\"seed\":7,\"lora_paths\":[\"$TMP/mf.safetensors\"]}")
  if [ "$code" = "400" ]; then ok "mage-flow" "adapter on a no-LoRA backend is a 400, not ignored"
  else bad "mage-flow" "returned $code for an adapter it cannot apply (want 400)"; fi
else skip "mage-flow" "not downloaded"; fi

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""

echo
echo "=================== RESULTS ==================="
printf '%-18s %-52s %s\n' "MODEL" "CHECK" "VERDICT"
for r in "${ROWS[@]}"; do
  IFS='|' read -r m c v <<<"$r"
  printf '%-18s %-52s %s\n' "$m" "$c" "$v"
done
echo "==============================================="
[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILURES"
exit $rc
