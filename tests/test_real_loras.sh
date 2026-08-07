#!/usr/bin/env bash
# REAL community LoRAs, downloaded from the Hub, asserting the renders are
# USABLE — not merely different.
#
# `test_multi_lora.sh` proves the stacking MATHS exactly (d+d == 2d, byte for
# byte) with synthetic adapters, and every one of its checks stayed green while
# a real adapter rendered pure static: its alpha lived in the safetensors
# `__metadata__` (PEFT) instead of a per-module `.alpha` tensor (kohya), so it
# ran at 1.0 = rank/alpha = 8x too strong. Bytes changed, deltas summed, noise.
#
# So this suite downloads what people actually publish and looks at the result.
# What each row uniquely covers:
#
#   klein-4B   Norod78 vintage-book-cover + old-gods — BFL `diffusion_model.
#              double_blocks.N.img_attn.qkv` keys, i.e. the FUSED-QKV fan-out
#              and third-split on a real file. No alpha anywhere (baked into
#              the weights) -> scale 1.0.
#   klein-9B   linoyts dreambooth + Delight — THE alpha fix. Both declare
#              alpha 4 / r 32 inside a JSON document held in one metadata
#              string; before the fix both ran at 1.0 and the dreambooth one
#              was static.
#   krea-2     gokaygokay Realism (flat `lora_alpha`/`lora_rank` metadata,
#              PEFT `base_model.model.` prefix) + krea darkbrush (NO metadata
#              at all, `transformer.` prefix, krea aliases like `img_in`).
#   ltx-2.3    joyfox Transition + Cseti CrossView-Warp — the video path,
#              `diffusion_model.` prefix, no metadata. (The plan named
#              Lightricks IC-LoRA-HDR; that repo is gated and 403s even with a
#              token, so an ungated IC-LoRA stands in.)
#
# MiniMax-H3's real adapter is the Turbo file shipped in its own pack, and
# `test_multi_lora.sh` already owns it (259/259 modules + a noise check on the
# render). H3 re-stages its whole pipeline per request, so it is not repeated
# here for zero new information.
#
# Adapters download once into ~/claude-tmp/lora-real/ and the renders are kept
# in ~/claude-tmp/lora-real/out/ — the metric is a tripwire, OPENING them is
# still the decisive check. A missing model or a failed download SKIPs.
#
# Usage: ./tests/test_real_loras.sh [port]
set -uo pipefail
PORT="${1:-11467}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
MODELS="${MLX_SERVE_MODELS:-$HOME/.mlx-serve/models}"
WORK="${LORA_REAL_DIR:-$HOME/claude-tmp/lora-real}"
OUT="$WORK/out"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
mkdir -p "$WORK" "$OUT"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-realora.XXXXXX")"
LOG="$WORK/server.log"
SRV=""
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT
rc=0
declare -a ROWS

row() { ROWS+=("$1|$2|$3"); }
ok()  { echo "  PASS: $2"; row "$1" "$2" "PASS"; }
bad() { echo "  FAIL: $2"; row "$1" "$2" "FAIL"; rc=1; }
skip(){ echo "  SKIP: $2"; row "$1" "$2" "SKIP"; }

have() { [ -d "$MODELS/$1" ]; }

# Backends run in turn on ONE server; without this each finished one stays
# resident and the next loads on top of it, which is how the sibling suite's
# last backend ended up refused by the media memory preflight with a 0.1%
# margin. Nothing here needs two backends resident at once.
unload() { curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/unload-model" \
  -H 'Content-Type: application/json' -d "{\"model\":\"$1\"}"; }

# fetch <repo> <file-in-repo> <local-name> → echoes the path, non-zero if the
# download failed. Already-present files are never re-fetched.
#
# Staged through `.partial` and renamed on success, which is `DownloadManager`'s
# rule and not tidiness: an interrupted download would otherwise leave a
# TRUNCATED safetensors that the next run treats as complete, and a truncated
# safetensors reaching `mlx_load_safetensors` raises an MLX error — which is
# not a Zig error, it kills the server. The `.partial` also keeps resume (-C -)
# working across runs.
fetch() {
  local dest="$WORK/$3" part="$WORK/$3.partial"
  [ -s "$dest" ] && { echo "$dest"; return 0; }
  echo "  downloading $1/$2 ..." >&2
  if curl -sfL -C - -o "$part" "https://huggingface.co/$1/resolve/main/$2"; then
    mv "$part" "$dest" && { echo "$dest"; return 0; }
  fi
  return 1
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

post() {
  curl -s --max-time 3000 -o "$2" -w '%{http_code}' -X POST \
    "http://127.0.0.1:$PORT/$1" -H 'Content-Type: application/json' -d "$3"
}

# usable <label> <response.json> <what> <artifact.png> → not static, and kept.
usable() {
  local m rc2
  m=$(python3 "$ROOT/tests/lora_noise.py" "$2" --save "$OUT/$4" 2>&1); rc2=$?
  case $rc2 in
    0) ok   "$1" "$3 renders (adjacent-pixel delta $m)"; return 0 ;;
    3) skip "$1" "$3 noise check: $m"; return 0 ;;
    *) bad  "$1" "$3 is NOISE ($m, bar 20) — see $OUT/$4"; return 1 ;;
  esac
}

logfrom() { awk -v n="$1" 'NR>n' "$LOG"; }

# The engine logs the scale it RESOLVED for each file. An adapter running at a
# scale nobody declared is the whole bug, so the number is asserted, not just
# printed.
check_scale() { # <label> <line-mark> <basename> <expected 4dp>
  local got
  got=$(logfrom "$2" | grep -o "\[lora\] $3: scale [0-9.]*" | tail -1 | awk '{print $NF}')
  if [ "$got" = "$4" ]; then ok "$1" "$3 runs at its declared scale ($4)"
  else bad "$1" "$3 scale is '${got:-<not logged>}', want $4"; fi
}

check_attached() { # <label> <line-mark> <expected adapter count>
  local line n adapters
  line=$(logfrom "$2" | grep -o 'lora: matched [0-9]* module-attachment(s) across [0-9]* adapter(s)' | tail -1)
  n=$(echo "$line" | awk '{print $3}')        # "lora: matched N module-attachment(s) across M adapter(s)"
  adapters=$(echo "$line" | awk '{print $6}')
  if [ -n "$n" ] && [ "$n" -gt 0 ] && [ "$adapters" = "$3" ]; then
    ok "$1" "$3 adapter(s) attached, $n module-attachment(s)"
  else
    bad "$1" "attach line wrong: '${line:-<none>}' (want $3 adapters, >0 modules)"
  fi
}

boot || exit 1

# ════════════════════════════════════════════════════════════════════════
# One backend: baseline → A alone → A+B stacked. Every render must be a
# picture, and each step must visibly move from the one before it.
# ════════════════════════════════════════════════════════════════════════
real_matrix() { # <label> <endpoint> <model> <A> <A-scale> <B> <B-scale> <extra-json>
  local label="$1" ep="$2" model="$3" A="$4" as="$5" B="$6" bs="$7" extra="$8"
  echo "== $label"
  local base="{\"model\":\"$model\",\"seed\":7,$extra"
  local code mark

  code=$(post "$ep" "$TMP/$label.base.json" "$base}")
  [ "$code" = "200" ] || { bad "$label" "baseline (http $code: $(head -c 160 "$TMP/$label.base.json"))"; return; }
  usable "$label" "$TMP/$label.base.json" "baseline" "$label-1-baseline.png"

  mark=$(wc -l < "$LOG")
  code=$(post "$ep" "$TMP/$label.a.json" "$base,\"lora_paths\":[\"$A\"]}")
  if [ "$code" = "200" ]; then
    usable "$label" "$TMP/$label.a.json" "adapter A" "$label-2-A.png"
    check_scale "$label" "$mark" "$(basename "$A")" "$as"
    check_attached "$label" "$mark" 1
    if cmp -s "$TMP/$label.base.json" "$TMP/$label.a.json"; then
      bad "$label" "adapter A did not change the render"
    else
      ok "$label" "adapter A changes the render"
    fi
  else
    bad "$label" "adapter A failed (http $code: $(head -c 160 "$TMP/$label.a.json"))"
  fi

  mark=$(wc -l < "$LOG")
  code=$(post "$ep" "$TMP/$label.ab.json" "$base,\"lora_paths\":[\"$A\",\"$B\"]}")
  if [ "$code" = "200" ]; then
    usable "$label" "$TMP/$label.ab.json" "A+B stacked" "$label-3-AB.png"
    check_scale "$label" "$mark" "$(basename "$B")" "$bs"
    check_attached "$label" "$mark" 2
    # Two REAL adapters stacked must still be a picture AND must differ from
    # both inputs — the failure mode here is not "no effect", it is two deltas
    # piling up past what the model can render.
    if ! cmp -s "$TMP/$label.a.json" "$TMP/$label.ab.json" \
       && ! cmp -s "$TMP/$label.base.json" "$TMP/$label.ab.json"; then
      ok "$label" "stacking B on A changes the render again"
    else
      bad "$label" "stacked render matches A alone or the baseline"
    fi
  else
    bad "$label" "stacked pair failed (http $code: $(head -c 160 "$TMP/$label.ab.json"))"
  fi

  unload "$model"
}

IMG='"prompt":"a photo of a cat sitting on a chair","size":"512x512","steps":4'

# ── FLUX.2 klein-4B: real BFL fused-QKV keys ─────────────────────────────
if have "Runpod/FLUX.2-klein-4B-mflux-4bit"; then
  A=$(fetch Norod78/flux2-klein-4b-base-lora-vintage-book-cover \
            flux2-klein-4b-base-lora-vintage-book-cover.safetensors norod-vintage-book-cover.safetensors)
  B=$(fetch Norod78/flux2-klein-4b-lora-old-gods \
            flux2-klein-4b-lora-old-gods.safetensors norod-old-gods.safetensors)
  if [ -n "$A" ] && [ -n "$B" ]; then
    real_matrix "klein-4b" "v1/images/generations" "Runpod/FLUX.2-klein-4B-mflux-4bit" \
      "$A" "1.0000" "$B" "1.0000" "$IMG"
  else skip "klein-4b" "adapter download failed"; fi
else skip "klein-4b" "model not downloaded"; fi

# ── FLUX.2 klein-9B: the alpha fix itself ────────────────────────────────
if have "mlx-community/flux2-klein-9b-4bit"; then
  A=$(fetch linoyts/flux2-klein-lora pytorch_lora_weights.safetensors linoyts-dreambooth.safetensors)
  B=$(fetch linoyts/Flux2-Klein-Delight-LoRA pytorch_lora_weights.safetensors linoyts-delight.safetensors)
  if [ -n "$A" ] && [ -n "$B" ]; then
    # 4/32 — read out of the JSON document inside `lora_adapter_metadata`.
    # At the old 1.0 the first of these renders static, which is the
    # red-on-revert case for the whole suite.
    real_matrix "klein-9b" "v1/images/generations" "mlx-community/flux2-klein-9b-4bit" \
      "$A" "0.1250" "$B" "0.1250" "$IMG"
  else skip "klein-9b" "adapter download failed"; fi
else skip "klein-9b" "model not downloaded"; fi

# ── Krea-2-Turbo: flat metadata + a file with none at all ────────────────
if have "ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8"; then
  A=$(fetch gokaygokay/Krea-2-Realism-LoRA krea2_realism_lora.safetensors gokaygokay-krea2-realism.safetensors)
  B=$(fetch krea/Krea-2-LoRA-darkbrush darkbrush.safetensors krea-darkbrush.safetensors)
  if [ -n "$A" ] && [ -n "$B" ]; then
    # Realism declares alpha 32 / rank 32 = 1.0 — right by luck before the fix,
    # and still 1.0 after it, which is the point: the fix reads what the file
    # says rather than assuming.
    real_matrix "krea-2" "v1/images/generations" "ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8" \
      "$A" "1.0000" "$B" "1.0000" "$IMG"
  else skip "krea-2" "adapter download failed"; fi
else skip "krea-2" "model not downloaded"; fi

# ── LTX-2.3: the video path ──────────────────────────────────────────────
if have "dgrauet/ltx-2.3-mlx-q4"; then
  A=$(fetch joyfox/LTX-2.3-Transition-LORA ltx2.3-transition.safetensors joyfox-ltx-transition.safetensors)
  B=$(fetch Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp \
            LTX2.3-22B_IC-LoRA-CrossView-Warp_v0.9_18000.safetensors cseti-ltx-crossview-warp.safetensors)
  if [ -n "$A" ] && [ -n "$B" ]; then
    real_matrix "ltx-2.3" "v1/video/generations" "dgrauet/ltx-2.3-mlx-q4" \
      "$A" "1.0000" "$B" "1.0000" \
      '"prompt":"a cat walks across a room","width":256,"height":256,"num_frames":9,"steps":2'
  else skip "ltx-2.3" "adapter download failed"; fi
else skip "ltx-2.3" "model not downloaded"; fi

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""

echo
echo "=================== RESULTS ==================="
printf '%-12s %-58s %s\n' "MODEL" "CHECK" "VERDICT"
for r in "${ROWS[@]}"; do
  IFS='|' read -r m c v <<<"$r"
  printf '%-12s %-58s %s\n' "$m" "$c" "$v"
done
echo "==============================================="
echo "renders kept in $OUT — open them, the metric only catches static"
[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILURES"
exit $rc
