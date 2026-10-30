#!/usr/bin/env bash
# test_qwen_image_lora.sh — Qwen-Image-2.1 runtime LoRAs on BOTH variants (Turbo + base) with a real CHARACTER and a
# real STYLE adapter. Per variant, same prompt + seed: no adapter twice is byte-identical; scale 0 == the control;
# each adapter at 1.0 moves the render past DIFF_MIN and stays a picture; 0.5 lands strictly between 0 and 1.0; the
# stack differs from either alone; an edit with the style adapter differs from the same edit without; `matched N
# module-attachment(s)` equals the lora_A tensors in the FILE (a fused gate_up counts twice), with no "attached X of Y"
# and no `unsupported lora format`. An eight-tile contact sheet per variant is kept for opening by eye.
#
#   ./tests/test_qwen_image_lora.sh [port]   # QWEN_LORA_DIR, QWEN_LORA_STEPS (30), DIFF_MIN (2.0)
#
# Adapters (downloaded once): bghira/cog-qwen-image-2.1-domokun-lora (trigger 🟫, license other: local testing only),
# wesjos/Qwen-Image2.1-ZenlessZoneZero-Lora v1.7_4000 (`zenlesszonezero`, apache-2.0, a fused img_mlp.gate_up).
# SKIPs a variant whose pack is absent.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11471}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
MODELS="${MLX_SERVE_MODELS:-$HOME/.mlx-serve/models}"
WORK="${QWEN_LORA_DIR:-$HOME/claude-tmp/qwen-lora}"
OUT="$WORK/out"
STEPS="${QWEN_LORA_STEPS:-30}"
DIFF_MIN="${DIFF_MIN:-2.0}"
M="$ROOT/tests/qwen_image_lora_metrics.py"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
mkdir -p "$WORK" "$OUT"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-qwenlora.XXXXXX")"
LOG="$WORK/server.log"
SRV=""
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT
rc=0
declare -a ROWS
row()  { ROWS+=("$1|$2|$3"); }
ok()   { echo "  PASS: $2"; row "$1" "$2" "PASS"; }
bad()  { echo "  FAIL: $2"; row "$1" "$2" "FAIL"; rc=1; }
skip() { echo "  SKIP: $2"; row "$1" "$2" "SKIP"; }

# Staged through `.partial` so an interrupted download never leaves a truncated
# safetensors that the next run trusts (a truncated file kills the server).
fetch() { # <repo> <file-in-repo> <local-name>
  local dest="$WORK/$3" part="$WORK/$3.partial"
  [ -s "$dest" ] && { echo "$dest"; return 0; }
  echo "  downloading $1/$2 ..." >&2
  curl -sfL -C - -o "$part" "https://huggingface.co/$1/resolve/main/$2" && mv "$part" "$dest" && { echo "$dest"; return 0; }
  return 1
}

CHAR=$(fetch bghira/cog-qwen-image-2.1-domokun-lora pytorch_lora_weights.safetensors char-domokun.safetensors) \
  || { echo "FAIL: character adapter download"; exit 1; }
STYLE=$(fetch wesjos/Qwen-Image2.1-ZenlessZoneZero-Lora zzz_style_v1.7_4000.safetensors style-zzz.safetensors) \
  || { echo "FAIL: style adapter download"; exit 1; }
python3 "$M" expected "$CHAR" >/dev/null 2>&1; [ $? -eq 3 ] && { echo "SKIP: numpy+PIL missing"; exit 0; }
EXP_CHAR=$(python3 "$M" expected "$CHAR")
EXP_STYLE=$(python3 "$M" expected "$STYLE")

boot() {
  "$BIN" --serve --port "$PORT" --model-dir "$MODELS" >"$LOG" 2>&1 &
  SRV=$!
  srv_wait_health "http://127.0.0.1:$PORT" 120 0.5 "$SRV" && return 0
  kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server died"; tail -5 "$LOG"; return 1; }
  echo "FAIL: server never became healthy"; return 1
}
post() { curl -s --max-time 3600 -o "$2" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/$1" -H 'Content-Type: application/json' -d "$3"; }
unload() { curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/unload-model" -H 'Content-Type: application/json' -d "{\"model\":\"$1\"}"; }
logfrom() { awk -v n="$1" 'NR>n' "$LOG"; }

ON='a photograph of 🟫 waving, zenlesszonezero style'
OFF='a photograph of a plush figure waving'

variant() { # <label> <model id> <extra json fields>
  local label="$1" model="$2" extra="$3"
  echo "== $label ($model)"
  local d="$TMP/$label"; mkdir -p "$d"
  local common="\"model\":\"$model\",\"seed\":7,\"size\":\"512x512\",$extra"
  local gen_ep="v1/images/generations" code mark

  # gen <name> <prompt> <lora_paths json> <lora_scales json> → $d/<name>.json, echoes http code
  gen() {
    local body="{$common,\"prompt\":\"$2\""
    [ -n "$3" ] && body="$body,\"lora_paths\":$3,\"lora_scales\":$4"
    post "$gen_ep" "$d/$1.json" "$body}"
  }
  need() { # <name> <prompt> <paths> <scales> → 0 when a render landed
    code=$(gen "$1" "$2" "$3" "$4")
    [ "$code" = 200 ] && return 0
    bad "$label" "$1: http $code ($(head -c 200 "$d/$1.json"))"; return 1
  }

  # 1 control
  need base "$ON" "" "" || return
  need base2 "$ON" "" "" || return
  if cmp -s "$d/base.json" "$d/base2.json"; then ok "$label" "1 control: no adapter twice is byte-identical"
  else bad "$label" "1 control: same seed rendered two different images (the metrics below mean nothing)"; fi
  python3 "$M" valid "$d/base.json" >/dev/null || bad "$label" "baseline is not a usable picture"

  # 2 scale 0
  need char0 "$ON" "[\"$CHAR\"]" "[0]" || return
  need style0 "$ON" "[\"$STYLE\"]" "[0]" || return
  if cmp -s "$d/base.json" "$d/char0.json" && cmp -s "$d/base.json" "$d/style0.json"; then
    ok "$label" "2 scale 0: both adapters byte-identical to the control"
  else bad "$label" "2 scale 0: an adapter at scale 0 moved the render"; fi

  # 3 effect (+ 7 coverage on the same runs)
  adapter_run() { # <name> <paths> <scales> <expected attachments> <adapter count>
    mark=$(wc -l < "$LOG")
    need "$1" "$ON" "$2" "$3" || return 1
    local v dm line n
    if v=$(python3 "$M" valid "$d/$1.json"); then :; else bad "$label" "3 $1 is not a usable picture ($v)"; fi
    dm=$(python3 "$M" diff "$d/base.json" "$d/$1.json")
    if awk -v d="$dm" -v t="$DIFF_MIN" 'BEGIN{exit !(d>t)}'; then ok "$label" "3 $1 moves the render (mean abs diff $dm > $DIFF_MIN; $v)"
    else bad "$label" "3 $1 barely changes the render (mean abs diff $dm <= $DIFF_MIN)"; fi
    eval "DIFF_$1=$dm"
    line=$(logfrom "$mark" | grep -o 'lora: matched [0-9]* module-attachment(s) across [0-9]* adapter(s)' | tail -1)
    n=$(echo "$line" | awk '{print $3}')
    if [ "$n" = "$4" ] && [ "$(echo "$line" | awk '{print $6}')" = "$5" ]; then ok "$label" "7 $1 coverage: matched $n = file's $4"
    else bad "$label" "7 $1 coverage: '${line:-<no attach line>}', want $4 across $5"; fi
    if logfrom "$mark" | grep -q -E 'attached [0-9]+ of [0-9]+ adapter module|unsupported lora format'; then
      bad "$label" "7 $1: partial attach or unsupported-format line in the log"
    fi
  }
  adapter_run char1 "[\"$CHAR\"]" "[1.0]" "$EXP_CHAR" 1 || return
  adapter_run style1 "[\"$STYLE\"]" "[1.0]" "$EXP_STYLE" 1 || return

  # 4 scale
  need char05 "$ON" "[\"$CHAR\"]" "[0.5]" || return
  need style05 "$ON" "[\"$STYLE\"]" "[0.5]" || return
  for a in char style; do
    local d1 d05
    eval "d1=\$DIFF_${a}1"
    d05=$(python3 "$M" diff "$d/base.json" "$d/${a}05.json")
    if awk -v lo=0 -v m="$d05" -v hi="$d1" 'BEGIN{exit !(m>lo && m<hi)}'; then ok "$label" "4 $a scale: abs diff @0=0 < @0.5=$d05 < @1.0=$d1"
    else bad "$label" "4 $a scale: @0.5 diff $d05 is not strictly between 0 and @1.0 diff $d1"; fi
  done

  # 5 stack
  adapter_run both "[\"$CHAR\",\"$STYLE\"]" "[1.0,1.0]" "$((EXP_CHAR + EXP_STYLE))" 2 || return
  if ! cmp -s "$d/both.json" "$d/char1.json" && ! cmp -s "$d/both.json" "$d/style1.json"; then ok "$label" "5 stack: differs from either adapter alone"
  else bad "$label" "5 stack: equals one of its parts"; fi

  # 6 edit — the control render is the source; same edit with and without the style adapter
  local src; src=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["b64_json"])' "$d/base.json")
  local ebody="{$common,\"mode\":\"edit\",\"prompt\":\"make it a night scene\",\"image\":\"$src\""
  local c1 c2
  c1=$(post "$gen_ep" "$d/edit0.json" "$ebody}")
  c2=$(post "$gen_ep" "$d/edit1.json" "$ebody,\"lora_paths\":[\"$STYLE\"],\"lora_scales\":[1.0]}")
  if [ "$c1" = 200 ] && [ "$c2" = 200 ]; then
    if python3 "$M" valid "$d/edit1.json" >/dev/null && ! cmp -s "$d/edit0.json" "$d/edit1.json"; then
      ok "$label" "6 edit: style adapter changes the edit (mean abs diff $(python3 "$M" diff "$d/edit0.json" "$d/edit1.json"))"
    else bad "$label" "6 edit: adapter did not change the edit, or it is not a picture"; fi
  else bad "$label" "6 edit: http $c1 / $c2 ($(head -c 160 "$d/edit1.json"))"; fi

  # contact sheet: row 1 WITH triggers (already rendered), row 2 the same seed and adapters WITHOUT them
  need base_off "$OFF" "" "" || return
  need char_off "$OFF" "[\"$CHAR\"]" "[1.0]" || return
  need style_off "$OFF" "[\"$STYLE\"]" "[1.0]" || return
  need both_off "$OFF" "[\"$CHAR\",\"$STYLE\"]" "[1.0,1.0]" || return
  python3 "$M" sheet "$OUT/sheet-$label.png" \
    "no LoRA (trigger prompt)=$d/base.json" "character (trigger)=$d/char1.json" \
    "style (trigger)=$d/style1.json" "character+style (trigger)=$d/both.json" \
    "no LoRA (no trigger)=$d/base_off.json" "character (no trigger)=$d/char_off.json" \
    "style (no trigger)=$d/style_off.json" "character+style (no trigger)=$d/both_off.json" \
    && echo "  sheet: $OUT/sheet-$label.png"

  unload "$model"
}

boot || exit 1

TURBO=$(ls -d "$MODELS"/ddalcu/Qwen-Image-2.1-Turbo-MLX-Serve-* 2>/dev/null | head -1)
BASE=$(ls -d "$MODELS"/ddalcu/Qwen-Image-2.1-MLX-Serve-* 2>/dev/null | head -1)
# Turbo runs its own sample_sigmas grid whatever `steps` says.
if [ -n "$TURBO" ]; then variant "turbo" "ddalcu/$(basename "$TURBO")" "\"steps\":4"; else skip turbo "pack not downloaded"; fi
if [ -n "$BASE" ];  then variant "base"  "ddalcu/$(basename "$BASE")"  "\"steps\":$STEPS"; else skip base "pack not downloaded"; fi

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
echo; echo "=================== RESULTS ==================="
printf '%-8s %-78s %s\n' VARIANT CHECK VERDICT
for r in "${ROWS[@]}"; do IFS='|' read -r v c s <<<"$r"; printf '%-8s %-78s %s\n' "$v" "$c" "$s"; done
echo "==============================================="
echo "contact sheets in $OUT — open them; the metrics cannot tell 'the character' from 'something changed'"
[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILURES"
exit $rc
