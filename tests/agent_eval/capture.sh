#!/bin/bash
# Build each run's web project and screenshot it over time, for judge.py.
#
#   tests/agent_eval/capture.sh <task> [run dir ...]     default: every run under out/<task>/
#
# Per run: `npm run build` (its result is recorded; a failed build is rendered with `vite build`
# alone and marked build_failed.txt), serve dist/ as static files, open it in headless Chrome at
# 1280x800, and take the task's frames (rubric.json "capture") into <run>/frames/f0.png, f1.png, ...
# Needs node, python3 and agent-browser (npm i -g agent-browser). Skips runs already captured.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TASK=${1:?usage: capture.sh <task> [run dir ...]}
shift
RUBRIC=$HERE/tasks/$TASK/rubric.json
read -r FRAMES INTERVAL SETTLE < <(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["capture"]
print(c["frames"], int(c["interval_s"] * 1000), int(c["settle_s"] * 1000))' "$RUBRIC")
JOBS=${JOBS:-4}
runs=("$@")
[ ${#runs[@]} -gt 0 ] || runs=("${OUT:-$HERE/out}/$TASK"/*/)

capture() {  # capture <run dir> <port>
  local run=${1%/} port=$2 p=${1%/}/project f=${1%/}/frames
  [ -f "$f/f$((FRAMES - 1)).png" ] && return
  [ -f "$p/package.json" ] || { echo "no project: $run"; return; }
  mkdir -p "$f"
  if ! (cd "$p" && npm run build > "$f/build.log" 2>&1); then
    echo "project build failed; rendered with vite build alone" > "$f/build_failed.txt"
    (cd "$p" && npx vite build > /dev/null 2>&1) || { echo "BUILD FAIL $run"; return; }
  fi
  python3 -m http.server "$port" --bind 127.0.0.1 --directory "$p/dist" > /dev/null 2>&1 &
  local srv=$! up=0
  for i in $(seq 1 30); do curl -sf "127.0.0.1:$port" > /dev/null && { up=1; break; }; sleep 1; done
  if [ $up = 0 ]; then  # never hand Chrome's error page to the judge as the scene
    echo "SERVE FAIL $run"; kill $srv; return
  fi
  local s="ae$port"
  agent-browser --session $s open "http://127.0.0.1:$port/" > /dev/null
  agent-browser --session $s set viewport 1280 800 > /dev/null
  agent-browser --session $s reload > /dev/null
  agent-browser --session $s wait "$SETTLE" > /dev/null
  for i in $(seq 0 $((FRAMES - 1))); do
    [ "$i" -gt 0 ] && agent-browser --session $s wait "$INTERVAL" > /dev/null
    agent-browser --session $s screenshot "$f/f$i.png" > /dev/null
  done
  agent-browser --session $s close > /dev/null 2>&1
  kill $srv
  echo "ok $run"
}

i=0
for run in "${runs[@]}"; do
  capture "$run" $((5400 + i % JOBS)) &
  i=$((i + 1))
  [ $((i % JOBS)) -eq 0 ] && wait
done
wait
