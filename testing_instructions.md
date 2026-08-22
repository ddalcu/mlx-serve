# Testing instructions — spec-decode width on your hardware

mlx-serve no longer needs a per-chip table row from you. The speculative-decode
width (MTP draft depth, DFlash/DSpark block) is **measured on your machine while
it serves**: every round's cost and yield goes into a per-(chip, model, quant,
OS) table under `~/.mlx-serve/round-cost/`, the plan reads it, and the file
survives reboots. A chip with no hand-typed row starts from a default cap and
converges on its own; the few rows that exist (M1 Pro, base M4/M5, M3 Ultra)
only shape the FIRST boot.

What is still worth sending from hardware we do not have: proof that it
converges there, and the before/after. That is ~10 minutes.

---

## Setup

Checkout github.com/ddalcu/mlx-serve  chore/fixes-lfm-dflash-auto-draft

Any checkpoint with an MTP head works (a Qwen3.8 pack), or one with a DFlash /
DSpark `drafter/` subdirectory (LFM2.5). A 27B on a 64 GB+ machine, something
smaller otherwise.

```bash
MODEL=/path/to/your/model          # e.g. a Qwen3.8 pack with an MTP head
BIN=./zig-out/bin/mlx-serve        # built with: zig build -Doptimize=ReleaseFast

serve() {  # serve <extra flags...>
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port 11250 \
    --mtp --prefix-cache-entries 0 --log-level info "$@" > srv.log 2>&1 &
  until curl -s -m 2 http://127.0.0.1:11250/health >/dev/null; do sleep 3; done
}
stop() { kill $(lsof -ti tcp:11250) 2>/dev/null; while lsof -ti tcp:11250 >/dev/null; do sleep 1; done; }
```

Measure on **AC power** at high charge. Battery→AC has doubled decode mid-run on
a laptop before, which reads as a fake win.

---

## Test 1 — your measured round-cost table (2 minutes, everyone)

```bash
serve
# run any two or three requests against it, then:
grep -a "spec-stats\|cost table\|adaptive depth cap\|round-cost table\|DFlash drafter ready" srv.log
cat ~/.mlx-serve/round-cost/*.txt
stop
```

Paste both outputs. The server measures the cost of every draft width it
actually runs (ms per round and tokens per round, per KV bucket) and writes the
table under `~/.mlx-serve/round-cost/` at the end of each request; the next
boot restores it. The `[spec-stats]` line carries `table=<bucket>:wN:ms/tok/n`
and `width_trials=`, and `[mtp] adaptive depth cap N (<row>, default 6)` names
the per-silicon row that caps the FIRST boot (the table may plan above it once
a width is measured). `MLX_SERVE_MTP_COST_TABLE=0` is the control arm (the
old controller), `MLX_SERVE_ROUND_COST_PERSIST=0` disables the file.

## Test 2 — before/after on an echo prompt (5 minutes)

Same server flags, two boots: the table OFF (the old controller) and ON
(default). Three requests each; the ON boot's LAST request is the number that
matters (the first one may carry a trial block). Warm is the realistic state —
do not delete `~/.mlx-serve/round-cost` between boots.

```bash
cat > prompt.json <<'JSON'
{"model":"MODEL_ID","temperature":0,"max_tokens":300,"stream":false,
 "messages":[{"role":"user","content":"Repeat the following text back to me exactly, three times in a row:\n\nThe quick brown fox jumps over the lazy dog while the diligent engineer measures the throughput of a speculative decoder on a laptop that is plugged into the wall and not running on battery power."}]}
JSON
# replace MODEL_ID with the id from: curl -s localhost:11250/v1/models

for arm in off on; do
  stop; [[ $arm == off ]] && export MLX_SERVE_MTP_COST_TABLE=0 || unset MLX_SERVE_MTP_COST_TABLE
  serve
  for r in 1 2 3; do
    curl -s -X POST http://127.0.0.1:11250/v1/chat/completions \
      -H 'Content-Type: application/json' -d @prompt.json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print('$arm rep $r: %.2f tok/s' % d['timings']['predicted_per_second'])"
    grep -a "spec-stats" srv.log | tail -1
  done
done
stop
```

An echo prompt is deliberate: acceptance is high, so the width actually
matters. On ordinary prose both arms sit at a shallow depth and read the same.

## Test 3 — DFlash / DSpark block chooser (optional, opt-in feature)

Only with a drafter sidecar. The per-round block chooser is off by default;
run Test 2's loop with `MLX_SERVE_DFLASH_CHOOSER=1` as the ON arm (and without
it as OFF). Paste the `[dflash] width chooser:` lines and the `block_hist=`
field: that is where the chooser tells you which widths it tried and settled
on. A MoE target (LFM2.5-8B-A1B) is the known rough edge — report it either way.

---

## What to send

Add a comment to this PR https://github.com/ddalcu/mlx-serve/pull/252:

```
Chip:        Apple M2 Max          (sysctl -n machdep.cpu.brand_string)
macOS:       26.2                  (sysctl -n kern.osproductversion)
Model:       <repo/name>, <quant>
Power:       AC

Test 1:
  [mtp] adaptive depth cap .. (<row or default>)
  [spec-stats] ... width_trials=.. table=<2k:w4:../.., w5:../..
  ~/.mlx-serve/round-cost/<key>.txt contents

Test 2 (decode tok/s, echo prompt):
  off: rep1 .. rep2 .. rep3 ..     on: rep1 .. rep2 .. rep3 ..
  the on arm's last [spec-stats] line

Test 3 (if you ran it): on/off tok/s + the [dflash] width chooser lines
```

A loss is as useful as a win: the table is supposed to cost nothing where the
default was already right, and the cases where it does not are the ones we
cannot see from here.

---

## Optional: the control arms

- `MLX_SERVE_MTP_COST_TABLE=0` — the MTP plan reads only the fitted prior (the
  table still observes and persists). Same-boot A/B control for anything above.
- `MLX_SERVE_ROUND_COST_PERSIST=0` — no table file: every boot starts cold.
- `MLX_SERVE_DFLASH_CHOOSER=1` — opt-in: the DFlash/DSpark block is chosen per
  round from the same table (serial is a candidate). Report its
  `[dflash] width chooser:` lines and `block_hist=` with on/off numbers.
