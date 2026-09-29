#!/bin/bash
# Run a coding agent (pi) on a task against an mlx-serve you already started, N times.
#
#   tests/agent_eval/run.sh <task> <label> [url] [runs]
#   tests/agent_eval/run.sh voxel-pagoda mixed48 http://127.0.0.1:8080 2
#
# Each run gets a fresh out/<task>/<label>-<n>/ with project/ (what the agent built), sessions/
# (pi's JSONL), pi.out (its final answer) and wall.txt. Start the arm you want to compare, run this
# with a label for it, then the next arm; judge.py groups runs by label. Needs pi on PATH.
# pi's config comes from `mlx-serve launch pi --print` (MLX_SERVE, default the one on PATH), so the
# context window, output cap and compaction reserve follow what the server advertises.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TASK=${1:?usage: run.sh <task> <label> [url] [runs]}
LABEL=${2:?usage: run.sh <task> <label> [url] [runs]}
URL=${3:-http://127.0.0.1:8080}
RUNS=${4:-1}
OUT=${OUT:-$HERE/out}
THINKING=${THINKING:-high}
TIMEOUT_MIN=${TIMEOUT_MIN:-120}
MLX_SERVE=${MLX_SERVE:-mlx-serve}
PROMPT=$HERE/tasks/$TASK/prompt.txt
[ -f "$PROMPT" ] || { echo "no task prompt: $PROMPT" >&2; exit 1; }
model=$(curl -sf "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') \
  || { echo "no server at $URL" >&2; exit 1; }

for n in $(seq 1 "$RUNS"); do
  i=1; while [ -e "$OUT/$TASK/$LABEL-$i" ]; do i=$((i + 1)); done
  dir=$OUT/$TASK/$LABEL-$i
  mkdir -p "$dir/project" "$dir/sessions" "$dir/home"
  HOME="$dir/home" "$MLX_SERVE" launch pi --url "$URL" --model "$model" --print --no-start > /dev/null \
    || { echo "mlx-serve launch pi could not write the config" >&2; exit 1; }
  pi_dir=$dir/home/.mlx-serve/pi
  echo "== $TASK/$LABEL-$i  model=$model  url=$URL"
  start=$(date +%s)
  (cd "$dir/project" && PI_CODING_AGENT_DIR="$pi_dir" timeout "${TIMEOUT_MIN}m" \
    "$HERE/pi_rpc.py" "$PROMPT" --provider mlx --model "$model" --thinking "$THINKING" \
    --session-dir "$dir/sessions" > "$dir/pi.out" 2> "$dir/pi.err")
  rc=$?
  echo "wall_s=$(( $(date +%s) - start )) exit=$rc model=$model" | tee "$dir/wall.txt"
done
