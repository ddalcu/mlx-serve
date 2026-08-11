#!/usr/bin/env bash
# Vision ARCH-DISPATCH sweep: one image turn against every vision code path we
# support, asserting both a relevant answer and the encoder-engagement line that
# proves the right path ran. This is the guard for shared-path edits (the image
# decode branch, VisionPreproc, the encode dispatch) — a change that quietly
# routes one arch into another's encoder passes an answer-only check and fails
# here. The per-arch scripts (test_qwen_vision / test_muse_vision /
# test_gemma4_unified_vision / test_dense_vision / test_vision_moe_regression)
# stay the DEPTH coverage; this is breadth.
#
# EVERY present candidate runs, not just the first: the packs differ in ways the
# loader has to absorb (mlx-community's muse ships the tower DENSE bf16 under
# bare `vision_tower.` prefixes, ours ships it quantized under `model.`), and a
# per-arch representative cannot see that.
#
# Usage: tests/test_vision_all.sh [port] [arch-filter-substring]
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-11387}"
ONLY="${2:-}"
IMAGE="tests/fixtures/house.jpeg"
ROOTS=(
  "$HOME/.mlx-serve/models"
  "/Volumes/G Drive SSD/models"
  "/Volumes/G Drive SSD/models-dl"
  "$HOME/.lmstudio/models"
)

# arch-label | boot-log signature | candidate dirs (relative to a root)
# No gemma-3: mlx-community's pack ships SigLIP under HF names
# (vision_tower.vision_model.encoder.*), not the Gemma-4 names our loader reads,
# so it serves text-only. Never wired, not a regression.
ARCHS=(
  "gemma-siglip|Vision encoder: [0-9]+ layers, hidden=|mlx-community/gemma-4-e4b-it-4bit mlx-community/gemma-4-26b-a4b-it-4bit mlx-community/gemma-4-26B-A4B-it-qat-4bit mlx-community/gemma-4-31b-it-4bit"
  "gemma-unified|Vision encoder: Gemma 4 12B unified|mlx-community/gemma-4-12B-it-qat-4bit mlx-community/gemma-4-12b-it-4bit"
  "qwen3-vl|Vision encoder: Qwen3-VL ViT|mlx-community/Qwen3.5-0.8B-MLX-4bit mlx-community/Qwen3.5-4B-MLX-4bit"
  "muse-glimmer|Vision encoder: Muse-Glimmer ViT|ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit mlx-community/Muse-Glimmer-30B-4bit"
)

if [ ! -f "$IMAGE" ]; then echo "SKIP: fixture $IMAGE missing"; exit 0; fi
B64=$(base64 -i "$IMAGE")
# The Gemma-only preprocessed-pixel format, at the size the app emits (768^2
# float32 CHW). A patch-grid tower must refuse it, not die dereferencing SigLIP
# weights it does not have (live crash 2026-08-11).
XPIX=$(python3 -c "import base64,array;print(base64.standard_b64encode((array.array('f',[0.0])*(3*768*768)).tobytes()).decode())")

# Every present candidate, one per line (a repo may sit under any root).
find_models() {
  for cand in $1; do
    for root in "${ROOTS[@]}"; do
      [ -f "$root/$cand/config.json" ] && { echo "$root/$cand"; break; }
    done
  done
}

RESULTS=(); FAIL=0
for spec in "${ARCHS[@]}"; do
  LABEL="${spec%%|*}"; rest="${spec#*|}"
  SIG="${rest%%|*}"; CANDS="${rest#*|}"
  if [ -n "$ONLY" ] && [[ "$LABEL" != *"$ONLY"* ]]; then continue; fi
  FOUND=$(find_models "$CANDS")
  if [ -z "$FOUND" ]; then RESULTS+=("SKIP $LABEL (no checkpoint)"); continue; fi

  while IFS= read -r MODEL; do
  NAME="$LABEL/$(basename "$MODEL")"
  echo "== $NAME =="
  LOG=$(mktemp)
  pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
  ./zig-out/bin/mlx-serve --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 300); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  if ! curl -s "localhost:$PORT/health" >/dev/null 2>&1; then
    RESULTS+=("FAIL $NAME (server never came up)"); FAIL=1; tail -15 "$LOG"; kill "$SRV" 2>/dev/null; continue
  fi

  # A pack can carry a vision_config and still ship no tower we can load (or one
  # under another project's key names — mlx-community's gemma-3 SigLIP is
  # HF-named, not Gemma-4-named). The server then serves it text-only and says
  # so, which is honest: SKIP rather than fail a checkpoint for a capability it
  # never advertised. Only a model claiming vision has to prove it.
  if ! curl -s "localhost:$PORT/v1/models" | grep -q '"vision"'; then
    RESULTS+=("SKIP $NAME (serves text-only: no loadable vision tower)")
    echo "  SKIP: pack advertises no vision"
    kill "$SRV" 2>/dev/null; sleep 2; continue
  fi

  bad=""
  grep -qE "$SIG" "$LOG" || bad="$bad wrong-encoder"

  ANS=$(cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null
{"model":"m","max_tokens":48,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the main subject of this image? One word."},
 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$B64"}}]}]}
EOF
)
  echo "  -> $(echo "$ANS" | tr '\n' ' ' | cut -c1-90)"
  echo "$ANS" | grep -qiE "house|home|building|cottage" || bad="$bad wrong-answer"

  # A patch-grid tower must survive the Gemma-only pixel format.
  if [ "$LABEL" != "gemma-siglip" ] && [ "$LABEL" != "gemma-unified" ]; then
    cat > /tmp/xpix_req.json <<EOF
{"model":"m","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"image_url","image_url":{"url":"data:image/x-mlx-pixels;base64,$XPIX"}},
 {"type":"text","text":"explain this image"}]}]}
EOF
    curl -s --max-time 120 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @/tmp/xpix_req.json >/dev/null 2>&1
    curl -s "localhost:$PORT/health" | grep -q '"ok"' || bad="$bad died-on-x-mlx-pixels"
    grep -q "Dropping x-mlx-pixels" "$LOG" || bad="$bad no-refusal-logged"
  fi

  kill "$SRV" 2>/dev/null; sleep 2
  if [ -z "$bad" ]; then RESULTS+=("PASS $NAME"); else RESULTS+=("FAIL $NAME:$bad"); FAIL=1; echo "  FAIL:$bad"; fi
  done <<< "$FOUND"
done

echo
echo "== vision arch sweep =="
for r in "${RESULTS[@]}"; do echo "  $r"; done
[ "$FAIL" = "0" ] && echo "PASS: vision arch sweep" || echo "FAIL: vision arch sweep"
exit $FAIL
