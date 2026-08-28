#!/bin/zsh
# Same-boot qwen4_exp A/B harness (M4 Max numbers in docs/gotchas/engine-mlx.md;
# the M5 box re-measures with the same script).
#   tests/qwen4_ab.sh mtp <tag> [server flags...]     MTP vs serial per prompt, 3 reps interleaved, then MTP + plain concurrent
#   tests/qwen4_ab.sh batched <tag> [server flags...] serial 3 reps per prompt, then 2-/4-stream prose + 2-stream 8.5k aggregate
# Env: QWEN4_MODEL (pack), MLX_SERVE_BIN, QWEN4_AB_OUT (default ~/claude-tmp/qwen4-ab), QWEN4_AB_PORT.
# Engine env (MLX_SERVE_MTP_TRACE, MLX_SERVE_MTP_FORCE_DEPTH, MLX_SERVE_HC_FUSED, ...) passes through to the server.
setopt nonomatch
mode=$1; tag=$2; shift 2
D=${0:A:h}/fixtures/qwen4_ab
O=${QWEN4_AB_OUT:-$HOME/claude-tmp/qwen4-ab}; mkdir -p $O
PACK=${QWEN4_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit}
BIN=${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}
PORT=${QWEN4_AB_PORT:-11414}
LOG=$O/${mode}_$tag.log
pkill -f "mlx-serve --model.*--port $PORT"
for i in $(seq 1 60); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 2; done
flags=(--max-concurrent 4 --prefix-cache-entries 0 --log-level info)
[[ $mode == mtp ]] && flags+=(--mtp)
nohup $BIN --model $PACK --serve --host 127.0.0.1 --port $PORT $flags "$@" > $LOG 2>&1 &
for i in $(seq 1 200); do curl -s -m 2 localhost:$PORT/health >/dev/null && grep -q "Model ready" $LOG && break; sleep 3; done
one() { echo "$1" | curl -s -m 1800 localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(f\"{t['predicted_per_second']:.1f}/{d['usage']['completion_tokens']}\")"; }
agg() { python3 -c "
import sys,time; t0=float(sys.argv[1]); toks=sum(int(x.split('/')[1]) for x in sys.stdin.read().split()); print(f'wall_agg={toks/(time.time()-t0):.1f} tok/s')" "$1"; }
# body <prompt> <nonce> <enable_mtp>: the nonce varies by REP, never by arm.
body() { python3 -c "import sys,json; d=json.load(open(sys.argv[1])); d['enable_mtp']=sys.argv[3]=='true'; d['messages'][-1]['content']='nonce '+sys.argv[2]+' '+d['messages'][-1]['content']; print(json.dumps(d))" $D/$1.json "$2" "$3"; }
now() { python3 -c 'import time; print(time.time())'; }
if [[ $mode == mtp ]]; then
  for p in code prose prompt; do for rep in 1 2 3; do for arm in true false; do echo "$tag $p mtp=$arm rep$rep $(one "$(body $p $rep $arm)")"; done; done; done
  for rep in 1 2 3; do
    rm -f $O/m_*.out; t0=$(now)
    (one "$(body prose m$rep true)" > $O/m_1.out) & p1=$!; (one "$(body prose p$rep false)" > $O/m_2.out) & p2=$!; wait $p1 $p2
    echo "$tag mtp+plain concurrent rep$rep mtp=$(cat $O/m_1.out) plain=$(cat $O/m_2.out) $(cat $O/m_*.out | agg $t0)"
  done
  echo "mtp engagements: $(grep -c 'spec-stats\] mode=mtp' $LOG)"
else
  for p in code prose prompt; do for rep in 1 2 3; do echo "$tag $p serial rep$rep $(one "$(body $p $rep false)")"; done; done
  for n in 2 4; do for rep in 1 2 3; do
    rm -f $O/s_*.out; t0=$(now); pids=(); for i in $(seq 1 $n); do (one "$(body prose ${rep}x$i false)" > $O/s_$i.out) & pids+=($!); done; wait $pids
    echo "$tag prose x$n rep$rep per-stream $(cat $O/s_*.out | tr '\n' ' ') $(cat $O/s_*.out | agg $t0)"
  done; done
  for rep in 1 2 3; do
    rm -f $O/s_*.out; t0=$(now); pids=(); for i in 1 2; do (one "$(body prompt ${rep}L$i false)" > $O/s_$i.out) & pids+=($!); done; wait $pids
    echo "$tag prompt x2 rep$rep per-stream $(cat $O/s_*.out | tr '\n' ' ') $(cat $O/s_*.out | agg $t0)"
  done
  echo "batched engagements: $(grep -c 'gdn batched decode engaged' $LOG)"
fi
pkill -f "mlx-serve --model.*--port $PORT"
