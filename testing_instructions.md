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
| **M4 / Pro** | both (only M4 **Max** was measured) | 1, 2, 3 |
| **M5 Pro / Max / Ultra** | depth cap (base M5 is 4; the dies are their own rows) | 1, 2 |
| **anything newer** | everything | 1, 2, 3 |

Already measured, no need to re-run unless you think a row is wrong:
M4 Max (depth 6, block 5), M3 Ultra (block 8), M1 Pro (depth 4), M5 base (depth 4).

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

## Test 1 — your measured ladder (2 minutes, everyone)

```bash
serve
grep -a "spec-cost\|adaptive depth cap\|DFlash drafter ready" srv.log
curl -s http://127.0.0.1:11250/props | python3 -m json.tool | grep -A12 spec_cost
stop
```

Paste both outputs. This is the server measuring your machine's verify-forward
cost ladder at boot; it costs ~1-2 s and is cached afterwards.

## Test 2 — MTP depth cap

Forces each depth and reports decode tok/s. The best depth is your row.

```bash
cat > prompt.json <<'JSON'
{"model":"MODEL_ID","temperature":0,"max_tokens":300,"stream":false,
 "messages":[{"role":"user","content":"Write a detailed technical explanation of how a B-tree index works in a relational database, including insertion, splitting and range scans."}]}
JSON
# replace MODEL_ID with the id from: curl -s localhost:11250/v1/models

for d in 3 4 5 6 7 8; do
  stop; serve --mtp-depth $d
  for r in 1 2; do
    curl -s -X POST http://127.0.0.1:11250/v1/chat/completions \
      -H 'Content-Type: application/json' -d @prompt.json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print('depth $d rep $r: %.2f tok/s' % d['timings']['predicted_per_second'])"
  done
done
stop
```

Report the **best of the two reps** per depth. Ignore rep 1 if it is much slower
than rep 2 — that one paid the kernel compile.

## Test 3 — DFlash block size

Only if you have a DFlash/DSpark drafter sidecar (a `drafter/` subdirectory in
the model, or one passed with `--drafter`). Same idea, different knob:

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

Test 1 ladder:
  [spec-cost] measured verify ladder (ms/forward) 1:.. 2:.. ...
  [spec-cost] measured draft step .. ms/position
  [mtp] adaptive depth cap ..

Test 2 (decode tok/s, best of 2):
  depth 3: ..   4: ..   5: ..   6: ..   7: ..   8: ..

Test 3 (decode tok/s, best of 2, if you ran it):
  block 3: ..   4: ..   5: ..   6: ..   7: ..   8: ..
  accepted/round at the best block: ..
```

A row lands only where somebody measured it — we never interpolate between
chips, and we would rather have your six numbers than a guess.

---

## Optional: the two opt-in levers

Both are off by default because they **measured a loss** on an M4 Max. If you
want to check whether that holds on your chip, run Test 2 with the env set and
compare against the same test without it:

- `MLX_SERVE_SPEC_COST_KV=1` — scales the cost model with context length.
  Measured -2.7% at a 21k prompt on M4 Max: it makes deep drafts look cheaper,
  the controller extends further, and the extra positions do not get accepted.
- `MLX_SERVE_SPEC_COST_EV=1` — replaces the hand-fitted cost surface with the
  boot probe's own fit. The probe times a forward, but a round is a forward plus
  the draft steps, so it under-prices depth.
- `MLX_SERVE_SPEC_COST_PROBE=0` — turns the boot probe off entirely and restores
  the hand-typed tables. Use it as the control arm for anything above.

If either of the first two is a *win* on your machine, that is genuinely
interesting — say so, with both arms' numbers.
