# Testing instructions — spec-decode width, per chip

Two of mlx-serve's speculative-decode settings are **per-silicon measurements**,
not formulas: the MTP draft depth cap and the DFlash/DSpark block. Each row in
those tables is a number somebody measured on one machine. If your chip has no
row, you get a conservative default that is probably leaving speed on the table.

This page is how you measure your chip in ~15 minutes and send us the row.

---

## Chips we need numbers for

Run the tests marked for your chip and open a PR (or an issue) with the output.

| Chip | What is unknown | Run |
|---|---|---|
| **M1 / M1 Max / M1 Ultra** | both caps (only M1 **Pro** was measured, depth 4) | 1, 2, 3 |
| **M2 / Pro / Max / Ultra** | both caps — no rows at all | 1, 2, 3 |
| **M3 / Pro / Max** | depth cap; block only measured on M3 **Ultra** | 1, 2, 3 |
| **M3 Ultra** | depth cap (block row exists: 8) | 1, 2 |
| **M4 Pro** | both (M4 base and M4 Max are done) | 1, 2, 3 |
| **M5 Pro / Max / Ultra** | depth cap (base M5 is 4; the dies are their own rows) | 1, 2 |
| **anything newer** | everything | 1, 2, 3 |

Already measured, no need to re-run unless you think a row is wrong:
M4 Max (depth 6, block 5), M3 Ultra (block 8), M1 Pro (depth 4),
M5 base (depth 4), M4 base (depth 4).

---

## Setup

Checkout github.com/ddalcu/mlx-serve  chore/fixes-lfm-dflash-auto-draft

Any checkpoint with an MTP head works. A 27B on a 64 GB+ machine, something
smaller otherwise — the *shape* of the answer is what we want, not the tok/s.

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

## Test 2 — MTP depth cap

`--mtp-depth` is a **cap**, not a fixed depth: the controller picks a depth
within `[1, cap]` every round, and **no env pins it** — `MLX_SERVE_MTP_ADAPTIVE=0`
only swaps one controller for another, which still moves. So the cap can only be
measured on a prompt whose acceptance is high enough to push the controller
against it. That means an **echo** prompt (ask the model to reproduce a passage
verbatim). On ordinary prose the controller sits at depth ~2 whatever the cap is,
every depth reads the same, and the table is worthless — one tester's first
attempt produced byte-identical `[spec-stats]` at depths 3, 5, 6, 7 and 8.

Confirm from the `[spec-stats]` line that `avg_per_round` actually reaches the
depth you asked for. If it does not, the cap never bound and that row is not a
measurement of that depth.

```bash
# An ECHO prompt: acceptance is high, so the cap binds and the sweep means
# something. Do NOT use ordinary prose here (see above).
cat > prompt.json <<'JSON'
{"model":"MODEL_ID","temperature":0,"max_tokens":300,"stream":false,
 "messages":[{"role":"user","content":"Repeat the following text back to me exactly, three times in a row:\n\nThe quick brown fox jumps over the lazy dog while the diligent engineer measures the throughput of a speculative decoder on a laptop that is plugged into the wall and not running on battery power."}]}
JSON
# replace MODEL_ID with the id from: curl -s localhost:11250/v1/models

for d in 3 4 5 6 7 8; do
  stop; serve --mtp-depth $d
  for r in 1 2; do
    curl -s -X POST http://127.0.0.1:11250/v1/chat/completions \
      -H 'Content-Type: application/json' -d @prompt.json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print('depth $d rep $r: %.2f tok/s' % d['timings']['predicted_per_second'])"
  done
  grep -a "spec-stats" srv.log | tail -1     # avg_per_round must reach $d
done
stop
```

Report the **best of the two reps** per depth, and the `[spec-stats]` line with
it. Ignore rep 1 if it is much slower than rep 2 — that one paid the kernel
compile. If a `[spec-stats]` line shows a depth other than the one you asked
for, drop that row rather than sending it: the controller moved and the number
is not a measurement of that depth.

## Test 3 — DFlash block size

Only if you have a DFlash/DSpark drafter sidecar (a `drafter/` subdirectory in
the model, or one passed with `--drafter`). This knob is a real fixed width —
no controller moves it — but it only clamps **downward** against the block the
sidecar was trained at, so values above that are silently no-ops. The
`DFlash drafter ready (block_size=N...)` line in `srv.log` says what you got.

```bash
for b in 3 4 5 6 7 8; do
  stop; serve --draft-block-size $b
  for r in 1 2; do
    curl -s -X POST http://127.0.0.1:11250/v1/chat/completions \
      -H 'Content-Type: application/json' -d @prompt.json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print('block $b rep $r: %.2f tok/s' % d['timings']['predicted_per_second'])"
  done
  grep -a "spec-stats" srv.log | tail -1
done
stop
```

The `[spec-stats]` line matters as much as the tok/s: it shows how many drafts
were accepted. A wider block that accepts no more tokens is a loss even when it
looks cheap.

---

## What to send

Add a comment to this PR https://github.com/ddalcu/mlx-serve/pull/252:

```
Chip:        Apple M2 Max          (sysctl -n machdep.cpu.brand_string)
macOS:       26.2                  (sysctl -n kern.osproductversion)
Model:       <repo/name>, <quant>
Power:       AC

Test 1 table:
  [mtp] adaptive depth cap .. (<row>, default 6)
  [spec-stats] ... width_trials=.. table=<2k:w4:../.., w5:../..
  ~/.mlx-serve/round-cost/<key>.txt contents

Test 2 (decode tok/s, echo prompt, best of 2 or median of 5):
  depth 3: ..   4: ..   5: ..   6: ..   7: ..   8: ..
  avg_per_round per depth: ..      ext_rounds per depth: ..

Test 3 (decode tok/s, best of 2, if you ran it):
  block 3: ..   4: ..   5: ..   6: ..   7: ..   8: ..
  accepted/round at the best block: ..
```

A row lands only where somebody measured it — we never interpolate between
chips, and we would rather have your six numbers than a guess.

---

## Optional: the control arms

- `MLX_SERVE_MTP_COST_TABLE=0` — the MTP plan reads only the fitted prior (the
  table still observes and persists). Same-boot A/B control for anything above.
- `MLX_SERVE_ROUND_COST_PERSIST=0` — no table file: every boot starts cold.
- `MLX_SERVE_DFLASH_CHOOSER=1` — opt-in: the DFlash/DSpark block is chosen per
  round from the same table (serial is a candidate). Report its
  `[dflash] width chooser:` lines and `block_hist=` with on/off numbers.
