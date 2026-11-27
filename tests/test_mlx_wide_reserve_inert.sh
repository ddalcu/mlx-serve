#!/bin/bash
# test_mlx_wide_reserve_inert.sh — patches/mlx-metal-wide-reserve.patch changes nothing until a caller arms it: the
# same server binary on the unpatched libmlx and on the patched one (DYLD_LIBRARY_PATH picks the dylib; the server
# looks the reserve up by name, so it runs on either) serves the same bytes, with the same MLX memory, for each model
# given, the reserve never armed (MLX_SERVE_DSV41_PF_KEEP_WIDE=0). Runs alternate stock, patched, stock, patched: the
# MLX counters (active, peak, cache after the request) must equal the stock runs' (within their own spread + 1 MiB),
# the server's RSS within the stock runs' spread + 1%, and the answer (choices, logprobs included) byte for byte.
#
#   ./tests/test_mlx_wide_reserve_inert.sh <stock lib dir> <patched lib dir> <model dir> [<model dir>...]
#     <lib dir>: a staged lib/mlx/lib (libmlx.dylib, libmlxc.dylib, mlx.metallib); the patched one must export
#     mlx_metal_wide_reserve, the stock one must not. Env: SERVER (default zig-out/bin/mlx-serve), PROMPT_FILE (a
#     chat workload JSON whose first message is the prompt; default a fixed text), MAX_TOKENS (64), CTX (32768).
#   Needs a GPU and the models' memory (a DeepSeek-V4.1 pack needs its wired limit raised); one server at a time.

set -u
cd "$(dirname "$0")/.." || exit 1
source tests/lib/server.sh
[ $# -ge 3 ] || { sed -n 2,13p "$0"; exit 2; }
STOCK=$1 PATCHED=$2; shift 2
SERVER=${SERVER:-zig-out/bin/mlx-serve}
PORT=${PORT:-11397}
U=http://127.0.0.1:$PORT
mkdir -p "$HOME/claude-tmp" && W=$(mktemp -d "$HOME/claude-tmp/wide-inert.XXXXXX")
PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

[ -x "$SERVER" ] || { echo "no server binary $SERVER"; exit 2; }
srv_preclean "$PORT"
nm -gU "$PATCHED/libmlx.dylib" | grep -q mlx_metal_wide_reserve && ok "the patched libmlx exports the reserve" || fail "$PATCHED/libmlx.dylib has no mlx_metal_wide_reserve"
nm -gU "$STOCK/libmlx.dylib" | grep -q mlx_metal_wide_reserve && fail "$STOCK/libmlx.dylib is patched" || ok "the stock libmlx has no reserve"

/usr/bin/python3 -I - "$W/body.json" "${PROMPT_FILE:-}" "${MAX_TOKENS:-64}" <<'P'
import json, sys
out, wl, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = json.load(open(wl))["messages"][0]["content"] if wl else "Write a Zig function that parses a CSV line into fields, handling quoted fields, then explain each branch. " * 40
json.dump({"messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": 0, "seed": 1234, "logprobs": True, "top_logprobs": 2, "stream": False}, open(out, "w"))
P

# free_gb: the memory a load can take now (free + inactive + speculative + purgeable pages), GB
free_gb() {
  vm_stat | awk '/page size of/{pg=$8} /^Pages free/{f=$3} /^Pages inactive/{i=$3} /^Pages speculative/{s=$3} /^Pages purgeable/{p=$3} END{print int((f+i+s+p)*pg/1e9)}'
}
FREE0=$(free_gb)

# serve <tag> <lib dir> <model>: one server, one request; $W/<tag>.answer (the choices) and $W/<tag>.mem
serve() {
  local tag=$1 lib=$2 model=$3 h=$W/home-$1
  mkdir -p "$h/.mlx-serve"
  # the last server's memory back first (macOS returns a large wired set seconds after the exit)
  for _ in $(seq 1 300); do [ "$(free_gb)" -ge $((FREE0 - 20)) ] && break; sleep 1; done
  env HOME="$h" DYLD_LIBRARY_PATH="$lib" DYLD_PRINT_LIBRARIES=1 MLX_SERVE_DSV41_PF_KEEP_WIDE=0 MLX_SERVE_DSV4_DSPARK=0 \
    "$SERVER" --model "$model" --serve --host 127.0.0.1 --port $PORT --ctx-size "${CTX:-32768}" > "$W/$tag.log" 2>&1 < /dev/null &
  local pid=$!
  srv_wait_health "$U" 600 1 $pid || { echo "server $tag did not start"; tail -5 "$W/$tag.log"; srv_stop $pid; return 1; }
  curl -s -m 900 -H 'Content-Type: application/json' -d @"$W/body.json" "$U/v1/chat/completions" |
    /usr/bin/python3 -I -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["choices"], sort_keys=True))' > "$W/$tag.answer"
  # the counters once the server is quiet (its idle warm-up forwards done): two reads 2 s apart agree
  local m="" prev="" rss
  for _ in $(seq 1 60); do
    m=$(curl -s "$U/props" | /usr/bin/python3 -I -c 'import json,sys; m=json.load(sys.stdin)["memory"]; print(m["active_bytes"], m["peak_bytes"], m["cache_bytes"])')
    [ -n "$m" ] && [ "$m" = "$prev" ] && break; prev=$m; sleep 2
  done
  rss=$(ps -o rss= -p $pid | tr -d ' ')
  echo "$m $((rss * 1024))" > "$W/$tag.mem"
  srv_stop $pid
  # which libmlx the process loaded (dyld's list)
  grep -a "libmlx.dylib" "$W/$tag.log" | head -1 | sed "s|^|    $tag loaded: |"
  grep -aq "$lib/libmlx.dylib" "$W/$tag.log" || { echo "    $tag did not load $lib/libmlx.dylib"; return 1; }
  echo "    $tag: active peak cache rss = $(cat "$W/$tag.mem")"
}

for model in "$@"; do
  name=$(basename "$model")
  echo "== $name: stock, patched, stock, patched =="
  okrun=1
  for t in s1:$STOCK p1:$PATCHED s2:$STOCK p2:$PATCHED; do serve "$name-${t%%:*}" "${t#*:}" "$model" || okrun=0; done
  [ $okrun = 1 ] || { fail "$name: a server did not run"; continue; }
  a=$W/$name
  if cmp -s "$a-s1.answer" "$a-p1.answer" && cmp -s "$a-s1.answer" "$a-p2.answer" && cmp -s "$a-s1.answer" "$a-s2.answer" && [ -s "$a-s1.answer" ]; then
    ok "$name: the same answer bytes, logprobs included ($(wc -c < "$a-s1.answer") bytes)"
  else fail "$name: the answers differ"; fi
  if /usr/bin/python3 -I - "$a-s1.mem" "$a-s2.mem" "$a-p1.mem" "$a-p2.mem" <<'P'
import sys
s = [list(map(int, open(f).read().split())) for f in sys.argv[1:3]]
p = [list(map(int, open(f).read().split())) for f in sys.argv[3:5]]
bad = []
for i, name in enumerate(["active", "peak", "cache", "rss"]):
    lo, hi = min(x[i] for x in s), max(x[i] for x in s)
    slack = (hi - lo) + (max(hi // 100, 1 << 20) if name == "rss" else 1 << 20)
    for x in p:
        if not (lo - slack <= x[i] <= hi + slack): bad.append(f"{name} {x[i]} outside stock {lo}..{hi} +- {slack}")
    print(f"    {name}: stock {[x[i] for x in s]} patched {[x[i] for x in p]}")
if bad: print("    " + "; ".join(bad)); sys.exit(1)
P
  then ok "$name: the same MLX memory (active, peak, cache) and RSS"; else fail "$name: memory differs"; fi
done

echo "PASS=$PASS FAIL=$FAIL (logs $W)"
[ "$FAIL" = 0 ]
