# Auto-adjusting speculative draft width (replace the cost MODEL with a cost MEASUREMENT)

## The problem

Speculative decode picks a width every round: the MTP draft depth, the
DFlash/DSpark block. Getting it right is worth 15-25% of decode throughput, and
the right answer depends on chip x quant width x sidecar geometry x context
length x workload. We cannot fill that table by hand.

Today it is filled three ways, all wrong the same way:

| mechanism | what it is |
|---|---|
| `mtp.adaptiveDepthCapForMachine` | hand-typed chip rows (M1 Pro 4, M5 4, M4 base 4, else 6) |
| `dflash.blockCapForMachine` | hand-typed chip rows (M3 Ultra 8, M4 5, else 5) |
| `generate.MTP_EV_DEFAULT_COSTS` + 5 `MTP_EV_G17_*` | a hand-fitted cost surface, refit four times |
| `src/spec_cost.zig` (2026-08-21) | a boot probe that tried to replace the above |

The controller optimizes `expected_tokens / modeled_cost`. Acceptance is
MEASURED (`mtp_ev_accept` EMAs). Cost is MODELED, and every failure so far is a
wrong cost model: the probe times a forward, not a round (marginals 0.088 vs
hand 0.20); the kv term modeled attention as amortized when it is O(k*L)
(measured -2.7%); the DFlash cliff says block 7 where the sweep says 5.

## What the three boxes measured (2026-08-22)

All curl sweeps, temp 0, max_tokens 300, `--prefix-cache-entries 0`, echo
prompt for MTP, novel prompt for DFlash. Raw tables:
`~/claude-tmp/peer-sweeps/`.

MTP depth, tok/s (a/r):

| cap | M4 base, Qwen3.5-9B 6-bit | M1 Pro, Qwen3.5-9B 6-bit | M1 Pro, Qwen3.8-27B iQ-3.8 |
|---|---|---|---|
| 3 | 47.44 (3.00) | 41.34 (3.00) | 14.09 (3.00) |
| 4 | **55.50** (4.00) | **41.91** (3.96) | **14.10** (3.96) |
| 5 | 47.50 (4.96, ext 23) | 39.30 (5.00) | 11.61 (5.00) |
| 6-8 | 47.28 (4.96, ext 23) | 39.29 (5.00) | 11.61 (5.00) |

Round time = a/r / tok/s. Going from cap 4 to cap 5 adds 32 ms/round on both
9B arms and **150 ms/round on the 27B**. The boot ladder's forward marginal
(4->5) is 3.6 ms on M4 base and 16 ms on M1 Pro 9B, the head step 10 and 6 ms.
The rest is unattributed: sync, draft readout, sampling, graph build. On the
27B it is a verify-width cliff (mixed-width pack, no split-K lane at M=5),
not a sync, and the probe declared that pack "unusable curve" and fell back
to the hand row without logging a number.

DFlash block, tok/s (a/r), novel prompt:

| block | M4 base 1.2B bf16 | M4 base 2.6B bf16 | M4 base 8B-A1B 4b | M1 Pro 2.6B 8b | M1 Pro 8B-A1B 8b |
|---|---|---|---|---|---|
| serial | 40.13 | 17.39 | 87.03 | - | ~75 (gate tripped at b3) |
| 3 | 44.32 | 20.34 | 87.65 | 87.00 | 75.39 |
| 4 | 50.75 | 22.25 | 86.26 | 90.69 | **82.80** |
| 5 (cap) | 53.38 | 23.27 | 84.19 | **92.27** | 80.06 |
| 6 | 54.10 | **24.04** | 80.47 | 60.84 | 63.19 |
| 7 | 55.08 | 17.29 (gate off) | 78.80 | 65.94 | 57.15 |
| 8 | **56.30** | 17.22 (gate off) | 73.39 | 73.90 | 57.05 |

Five pack x chip cells, five different answers (8, 6, none, 5, 4). The 8B-A1B
on M4 base never runs DSpark at all: realized 1.75-1.79 against the 1.80 MoE
gate floor at every block. No per-chip row can serve this table; the block is
an acceptance question first and a cost question second.

## Two corrections to the earlier draft of this plan

1. **The sync is unmeasured on every box.** The "~14 ms sync" was a residual:
   M4 base round delta minus the boot probe's forward marginal minus the
   probe's head step. Subtracting probe marginals (documented as 10x
   under-priced) from live round times mixes two methodologies, and the
   residual also holds readout, sampling and graph build. Nothing logs the
   sync today (`[spec-stats]` and `/props` both omit it), so Step 0 below
   exists to make it a measurement.
2. **Pricing the sync is not the whole fix, and a Phase 1 miss does not
   falsify the plan.** It should move the 9B arms (32 ms/round, sync is a
   real fraction of it) and cannot move the 27B (150 ms/round is a width
   cliff). A Phase 1 miss on the 27B is evidence FOR measured per-model cost
   (Phase 2), not against it. Do not gate Phase 2 on Phase 1.

## How the controller actually loses today (read the code, not the tables)

`mtpEvPlanForAt` (generate.zig ~6021): m_lo is the single-chunk argmax; the
extension horizon loop compares `cond <= best_r * mc` where `mc` is the
marginal verify+draft cost only. `costs.sync` (0.01) enters
`mtpEvRoundCost(with_sync)`, which the horizon never calls. The measured
`mtp_ev_sync_ms` only throttles DRY exploration (`mtpExtDryThresholdFor`),
never a productive extension.

On an echo prompt at the default costs the horizon is closed by a hair even
unpriced (a=0.98: best_r*mc = 0.987 > 0.98), so every extension rides the
`m_hi == m_lo` EXPLORATION VALVE at tau 0.95, and near-perfect confidence
clears 0.95 every round. The valve was added deliberately (comment at ~6003:
"deliberately NO separate is-the-sync-worth-it gate ... starves exploration")
because a prior-fed gate blocked the first trial forever. It is right that a
trial must happen; it is wrong that it happens every round with its cost
unpriced. Phase 1 keeps the valve and rate-limits it.

## Implementation

### Step 0: instrument (hermetic, ships first)

Append `sync_ms=` and `round_ms=` (the live EMAs) to the MTP `[spec-stats]`
line. Every later A/B reads these; testers can paste them. Format stays
grep-compatible (fields appended after `runtime_disabled=`).

### Phase 1: the regime gate (DONE here, 2026-08-22; peers re-running)

Pricing the sync inside the fitted surface was tried first and measured a
-2% LOSS on the M4 Max: `per_pos_hi` (0.26) was fitted from realized echo
rounds and already covers most of the sync, so adding the measured sync on
top double-counted. Mixing measured ms into modeled units is the same trap
as the probe. Replaced by a unit-free measurement:

- `MtpRegime` (generate.zig): EMAs of ms per EMITTED token for the two round
  SHAPES the plan can take, a two-chunk plan (sync paid whether or not the
  extension fires) vs a single-chunk plan, keyed on m_lo (a shape observed
  at a new base depth reseeds; the warmup climb's depth-1..3 rounds read 19
  ms/tok against 11 and are not a comparison). Warmup rounds are excluded.
- `mtpRoundPlan`: when both shapes are measured and two-chunk is worse per
  token, the two-chunk plan runs one round in `MTP_REGIME_EXPLORE_PERIOD` (8)
  and the rest collapse to single-chunk; when two-chunk is better (or single
  is unmeasured) a single-chunk round is forced once per period so the other
  shape stays measured. The horizon math is untouched.
- Exploration period scales with the measured gap (`ceil(gap/0.01)`,
  clamped 8..64): M4 base measured two-chunk 26% worse and a fixed 1-in-8
  was a 2% structural drag that kept cap 5 at 52.3 vs cap 4's 54.6.
- Inter-round WALL clock, not the in-round stopwatch: per-round overhead
  outside the round is part of a wider shape's win (M4 Max @16k read
  two-chunk 3.5% worse interleaved while the arm measured +3.7%). 5% margin.
- Guard: a unit test drives the real predicates in a loop and asserts both
  shapes occur. v2's "try the unmeasured single at once" compared `one_m`
  against a `two_m` only a two-chunk round can set, pinned single forever,
  and caps 5/6 "passed" with cap-4 numbers (ext_rounds=0, no verdict line).
  An echo workload cannot tell that apart from a win by tok/s alone.
- `[spec-stats]` now carries `sync_ms round_ms two_ms_tok one_ms_tok`; the
  verdict logs once per flip: `[mtp] regime gate: two-chunk X ms/tok vs
  single Y ms/tok -> two-chunk 1-in-8|every round`. Kill switch
  `MLX_SERVE_MTP_REGIME=0`.

v5.3 (FINAL, generate.zig diff sha 43c1d28d…, code-identical on the M1 Pro
by stripped-comment hash 845e542d…): v5.2 regressed 1-3% at short context on
the M1 Pro with the verdict flipping 5-7 times per boot (the majority shape's
first observed round after a trial block is still elevated, pulling the ratio
inside the 5% margin), and its first-round drop cost one extra pre-verdict
extension (the 27B's -2.6% at trials=0). v5.3 = verdict hysteresis (a
standing "worse" flips only at ratio <= 1.0) + the first observed round
counts. M1 Pro 9B v5.3: caps 5/6 42.1-42.4 vs cap 4 41.5-42.0 (+1.2%, every
boot), one verdict line per REQUEST (the bar; a boot of 3 reps prints 3);
9B 16k gated 34.92 vs control 33.20 (+5.2%, 3 boots each, best-of and
median agree); 27B cap 5 13.93 = v3 parity.
M4 base v5.3: caps 5/6 54.80/54.75 vs cap 4 54.20 (+1%), long-gen 53.96 vs
control 47.15 (+14.4%), 16k 44.64 vs 39.51 (+13.0%); best version on every gated cell. M4 Max
v5.3: wash short and 16k (+3% over depth 4 at 16k).

v5.1 (sha c20ec170…): v4's trial gate was `idx % period < 2` with
the period recomputed from the EMAs every round; the block's own observation
moved it by one per round and trials chained (M4 base long-gen ext_rounds
7 -> 14 -> 7 across v3/v4/v5). Replaced by an explicit schedule
(`next_trial`/`trial_end` on the regime, period read once when a block
starts) and `[spec-stats]` now splits `ext_rounds = pre-verdict + 2*trials`
(`verdict_round=`, `trials=`). M4 base v5.1, 3 boots each rep 3, after 2
throwaway boots: cap 4 54.35, cap 5 53.81 (-1.0%), cap 6 54.01 (-0.6%);
long-gen cap 5 53.45 vs control 47.34 (+12.9%); 16k cap 6 44.37 vs 39.53
(+12.2%). A 22-round request never reaches a scheduled trial (verdict at
round 16, trials=0); only long generations exercise the cadence.

M1 Pro v3 (sha fe1c7eb0…): 9B cap 5/6 gated 41.76 vs cap 4 41.97 (-0.5%)
vs ungated 39.33 (+6.2%), serial 24.7. 27B iQ-3.8bpw: ungated cap 5/6
10.79 against serial 10.45 (speculation worth +3%), gated 13.93 vs cap 4
14.06 (-0.9%), +29% over the ungated controller; gap 95 vs 71 ms/tok,
period 34. 9B 16k: wash at 2 reps, and the verdict flipped in-run (single
48.1 then 29.2 ms/tok), the transition bias on a second box and model.

v4: transition rounds (shape differs from the previous round) are
NOT observed and trials run as 2-round blocks, because the minority shape was
only ever measured on transitions, where the verify-width change costs a
one-off ~5% (M4 Max @16k interleaved read two-chunk 13.65 vs single 12.9
ms/tok while the homogeneous arms measure 13.0 vs 13.2, so v3 throttled the
better shape there). v4 on M4 Max 27B 4bit: short echo 97.8-99.9 vs control
96.1-96.4 (+1.7%), 16k 77.7-78.0 vs 77.9-78.0 (wash, verdict "every round"
in 6/6 boots), depth-4 97.1 / 76.0.

M4 base v3, 15 boots: cap 4 median 54.64 (0.7% spread), cap 5 54.43, cap 6
53.43; all low readings are the first 4 boots (transient, tracks
ext_rounds). `drafted == 4*attempts + ext_rounds` held 15/15, so m_lo is 4
and m_hi 5 on that pack whatever the cap: caps 5 and 6 are behaviourally
identical there and a cap-6-wide extension is untested on that box.

Measured sync: M4 base 38.6 ms per two-chunk round (30% of 127 ms), M4 Max
5-8 ms. M4 base v1 (1-in-8): cap 5 -14% -> -4.2% vs cap 4 short, +12.5% over
control on long-gen, +5.2% at 16k.

Measured v1, M4 Max Qwen3.8-27B 4bit echo, same build, two boot orders, rep 3:
gate on 97.06/97.02, gate off 97.39/97.31, `--mtp-depth 4` 97.33/95.87
(two-chunk 10.66 vs single 10.24 ms/tok: a wash, throttled to 1-in-8, lands
between). Bar for the peers: M4 base cap 5/6 should stop losing 17% to cap 4
on the 9B; the 27B on M1 Pro is expected to move little (width cliff, Phase
2's motivation).

### Phase 2: the measured round-cost table (pure, hermetic)

`RoundCostTable` on `LoadedModel` (per MODEL; a Generator is per request and
its kv spans only `max_tokens`, already a live bug once):

```
min_ms[depth][kv_bucket]   single-chunk rounds only
seen[depth][kv_bucket]
```

- Buckets: `<2k, 2-4k, 4-8k, 8-16k, 16-32k, 32k+`. 6 x 8 floats.
- MIN, never mean: contention only ADDS time. Sample only when
  `spec_cost_solo`. Two-chunk rounds are a different measurement: the sync
  is measured separately (Phase 1), so store only single-chunk rounds here and
  price a two-chunk plan as `min_ms[m_hi] + sync_ms`.
- **MIN must age.** A MIN taken on a cool box is a lie after thermal soak and
  the controller then plans on a cost the GPU no longer delivers. Keep a
  windowed MIN (reset the bucket after N samples, or decay toward the latest
  sample slowly) and test that a sustained slower regime moves the estimate.
- Unmeasured widths extrapolate from the two nearest measured widths (cost is
  near-linear between cliffs). `MTP_EV_DEFAULT_COSTS` survives only as the
  cold-start prior.
- A rejected/unusable sample LOGS its numbers. The probe's silent
  "unusable curve" fallback is the trap this must not repeat.
- Exploration: the Phase 1 valve already probes `m_lo + 1` one round in N; a
  width rejected under a hard workload gets re-tried when the workload eases.

`mtpObserveKvCost` (generate.zig ~5227/5324) is already the per-round
`(m, kv_len, round_ms)` observation site; the table replaces the kv learner
it feeds.

### Phase 3: wire it, delete the rest

- `mtpEvMarginalCost*` / `mtpEvRoundCost*` read the table.
- Delete `src/spec_cost.zig`, `Transformer.probeSpecCostCurve`,
  `mtp.probeStepMs`, `spec_cost_curve`, the `/props spec_cost` object, the
  `MLX_SERVE_SPEC_COST_{PROBE,EV,KV,BLOCK}` gates.
- Delete `mtp.adaptiveDepthCapForMachine` and the five `MTP_EV_G17_*`
  surfaces. `MAX_DEPTH` (8) stays as a safety rail.
- Keep `--mtp-depth` / `MLX_SERVE_MTP_ADAPTIVE=0`: every A/B needs a forced
  width, and they are the escape hatch.

### Phase 4: DFlash/DSpark block chooses itself per round

The block is fixed per request today with a yield gate that can only turn
spec OFF. The table above shows the answer is per pack, per chip, and
acceptance-dominated, so:

- Per-round block selection from measured cost (the same `RoundCostTable`
  keyed on block) x measured per-position acceptance (the dflash EMAs the
  yield gate already keeps), resolved every round like MTP's depth, not at
  admission. Acceptance is not known at admission.
- The width controller SUBSUMES the yield gate: "block 0 / serial" is one of
  the candidates and wins when nothing pays. The M4 base 8B-A1B case (misses
  the 1.80 floor by 0.01-0.05 at every block, best arm 1.01x) is the test
  that the gate and the chooser agree.
- The downward clamp against the sidecar's trained `config.block_size` stays;
  `dflash.blockCapForMachine` goes.

### Phase 5: persist (only if measured cold start costs something)

`~/.mlx-serve/round-cost/<chip,model,quant,os>.json`, versioned, stale = quiet
miss. Do not build speculatively.

## Discipline for Phase 2+ (what the Phase 1 day cost, so it is not paid twice)

1. **Instrument first.** Three of five Phase 1 versions died for lack of a
   counter: `sync_ms` (the 14 ms residual vs the real 38.6), `verdict_round`
   and `trials` (the 14/72 trial chain). Every new controller term ships its
   `[spec-stats]` field and a one-shot verdict line in the SAME change, before
   any live run.
2. **A stateful gate gets a simulated-loop unit test before a live run.** The
   v2 deadlock passed every predicate test and read as a PASS on echo; the
   loop test (ask, run that shape, feed the observation) catches that class in
   one assertion. Phase 2's table chooser gets the same harness: drive it for
   200+ rounds with synthetic costs and assert every width it is allowed to
   pick gets measured.
3. **An identity the log can check beats a mechanism argument.**
   `drafted == 4*attempts + ext_rounds` refuted the m_lo hypothesis;
   `ext_rounds == pre-verdict + 2*trials` proved the v5 schedule. Design the
   counters so such identities exist.
4. **Testers pull a branch; diffs are not a delivery mechanism.** Five pasted
   patches in one day, two revert traps (`--3way` stages), one hash catch. A
   pushed branch plus the sha256 of the diff is the protocol.
5. **The short echo prompt is a smoke test, not the bar.** A 300-token echo
   is ~22 rounds: on M4 base the verdict forms at round 16 and no scheduled
   trial ever runs. The cells that exercise a controller are long-gen (70+
   rounds) and 16k, with 3+ boots per side (16k boot spread measured 2.6
   tok/s on M1 Pro, larger than the effect).
6. **Report every boot; label, do not discard.** The M4 base "first 3 boots"
   transient appeared in one session and not the next; mechanism unknown.
7. **Where hardcoding still wins today**: the DFlash block and the MTP
   verify-width cliff (M1 Pro 27B pays +150 ms/round at depth 5) are
   acceptance x cost cliffs per pack and chip. The swept rows stay as the
   floor until Phase 2/4 measure them, and `--mtp-depth`, `--draft-block-size`
   and the `MLX_SERVE_MTP_REGIME=0` switch stay as escape hatches.

## Verification

1. Phase 1 on M4 Max (done, wash), then both peers re-run Test 2 on the
   regime-gate build with the `[spec-stats]` lines pasted.
2. No regression on measured machines: M4 Max (depth 6, block 5), M3 Ultra
   (block 8, unreachable; row stays), M1 Pro (4), M5 base (4), M4 base (4).
   The controller must FIND these numbers; that is the bar.
3. Long context 8k/16k/32k on all three boxes, ONE prompt builder shared
   across boxes (repeated filler to length + the echo tail), never swept
   before.
4. DFlash Phase 4 bar: the five-cell table above, each cell within 3% of its
   swept best, and the 8B-A1B M4 base cell not worse than serial.
5. `tests/test_mtp_equivalence.sh`, `tests/test_dflash.sh`,
   `tests/test_dspark_lfm2.sh` green WITH a model env set (they exit 0 and
   skip without one).

### Traps, all paid for already

- `--mtp-depth` is a CAP and no env pins a depth. Sweep on an ECHO prompt and
  confirm `avg_per_round` reaches the cap; the novel-prompt cells were flat
  because acceptance never cleared the 0.60 promote threshold.
- A skipped test reads as a pass.
- Keep `zig build -Doptimize=ReleaseFast` in the loop (lazy analysis hides
  compile errors under `zig build test`).
- An A/B arm is proven by an ENGAGEMENT line in its own log, never its launch
  env.
- Spec cells are variance: medians, counterbalanced order, same-boot ratios,
  AC power, corroborate with `[spec-stats]` counters.
- Never bench against the user's live app server (port 11234); boot on a
  spare port, kill by port, never `pkill -f mlx-serve`.
- Serial baselines: every DFlash cell needs a spec-off arm beside it (the M1
  Pro sweep had none).

## The rule this is all an instance of

Throughput is accepted-tokens OVER round-cost. Anything that measures only
the denominator is wrong in the direction of drafting too much; anything that
measures only the numerator (the EV controller's valve on echo) is wrong the
same way. A hardcoded row is acceptable when someone measured both together
on real hardware, which is why the swept chip rows beat the probe today. The
fix is to let the controller measure the whole fraction itself, per model,
at the real context.

## State (2026-08-22 end of day)

Phase 1 DONE, uncommitted: `src/generate.zig` regime gate v5.2 (v5.1 +
`mtpRegimeForce` idempotent per round_idx; diff sha 80db1f2c…; v5.1
c20ec170… was the version measured on the M4 base box), `src/transformer.zig`
lfm2 torch-layout conv fix, content-identical to the M4 box's commit a962c38
(content hash c2d1f2b5… on both trees). Full suite 9/9, `test_mtp_equivalence.sh`
11/11 on the 27B. Results vs today's controller: M4 base 9B +12.9% long-gen,
+12.2% 16k, caps within 1% of cap 4; M1 Pro 27B +29% (ungated spec was worth
+3% over serial), 9B +6.2%; M4 Max wash. M1 Pro v5.1 confirmation pending its
user's clearance. Raw tables: `~/claude-tmp/peer-sweeps/`.

Branch `chore/fixes-lfm-dflash-auto-draft`, PR #252. Contains the boot probe,
the opt-in kv term, the opt-in fitted marginals, the m4-base row and
`testing_instructions.md`. The M4 box holds `5c4245a` (LiquidAI bf16 LFM2
depthwise conv layout fix) on a DIVERGED base (`4befe03`); it is applied here
uncommitted and must be rebased onto `7617003` and pushed from that box.
