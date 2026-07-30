# Next decode-perf levers — handoff

> **Update 2026-07-29 (fourth round — executing the third round's list).** All five
> items below were attempted; two shipped default-on, one ships opt-in after the live
> number refused to match the µbench, one dissolved on contact with the checkpoints,
> and one turned out to already be our construction. Laguna XS live decode is now
> **~8.3 ms/tok short-context (~121 tok/s)**. Full stories: `docs/gotchas/engine-mlx.md`
> ("Round 4").
>
> | item | result | switch |
> |---|---|---|
> | 1. Certified lm_head prune (`lmHeadPruneProject`, argmax-only requests via `ForwardCtx.argmax_only`) | **NULL live, ships OPT-IN.** Kernels sound (bound-soundness + candidate bit-identity pinned; live candidates median ~19/100352); µbench at the real geometry reads dense 1.04 → pruned 0.63 ms/iter, but the INTERLEAVED same-boot A/B (presence_penalty=1e-9 as the per-request dense arm, 20 rounds) reads **+0.84% median, 3/20 wins**. Where the ~0.5 ms goes live is unattributed — trace + µbench arms left in place | `MLX_SERVE_LMHEAD_PRUNE=1` opts in; `MLX_SERVE_LMHEAD_PRUNE_TRACE=1` counts candidates; `MLX_SERVE_LMHEAD_UBENCH=1` re-runs the 8-arm µbench |
> | 2. MoE down-path tail fusion (`gatherQmvDownReduce`: down-gather + router weighting + K-reduction, one dispatch) | **−1.5% Laguna XS, 2/2 pairs both orders (8.58/8.56 vs 8.71/8.70), BIT-IDENTICAL** (parity at small AND real [1,1,8,2048] geometry, nvfp4 + affine), default ON. Shared-expert/residual absorption + the residual+RMS+router fold deliberately deferred: more stock-kernel replicas for ~0.1–0.2 ms of mostly launch latency, same family as two measured nulls | `MLX_SERVE_MOE_DOWN_REDUCE_FUSED=0` |
> | 3. QK-norm+RoPE for qwen archs | **Dormant — every qwen3.5/3.6 checkpoint is hd 256** (2B/9B/27B/35B all checked; the third-round "hd 128 fits as-is" premise was wrong). rd=32 + `gatedFullAttnWith` wiring shipped and parity-pinned (incl. the post-split strided q view); engages automatically on a future hd-128 QK-norm arch. hd-256 lane mapping remains the real lever for qwen AND gemma4 | `MLX_SERVE_QK_NORM_ROPE_FUSED=0` (unchanged) |
> | 4. NVFP4-g16 tail for `--decode-attn-quant` (layers ≥ 80%) | **−3.3% Laguna XS, 2/2 pairs (8.25/8.26 vs 8.52/8.55), default ON** under the existing lossy toggle; 6-prompt greedy characterization re-run, answers unchanged in substance | `MLX_SERVE_DECODE_ATTN_QUANT_NVFP4_FROM=<layer>` or `off` |
> | 5a. Prefill sorted-MoE tail | **Already our construction** — the tail gathers activations through the inverse permutation (`take_axis(down, inv_order)`), no expert-bank copy exists to remove | n/a |
> | 5b. Prefill fused gate+up bank | **Still open** (load-time [E, 2·inter, hidden] concat, additive memory; A/B with `tests/prefill_ab.sh`) | — |
>
> Remaining open levers, in the order I would take them: the hd-256 QK-norm+RoPE lane
> mapping (now covers qwen3.5/3.6 + gemma4 — a much bigger surface than the third round
> believed); 5b above; the lm_head prune's live-cost attribution (the µbench says the
> win exists); the verify-lane `metal_kernel` config caching from the second round's
> item 5.

> **Update 2026-07-29 (third round — the mlxfast pull).** The mlxfast-challenge tree
> jumped from ~1.12 to ~1.385 in two days; we pulled it, mined every lever, and shipped
> three things. Laguna XS live decode is now **~8.8 ms/tok short-context** (~113 tok/s).
>
> | change | result | switch |
> |---|---|---|
> | `--decode-attn-quant` — INT8-g32 side copies of dense bf16 attention weights, decode+verify only (mlxfast NATIVE_AFFINE class, ~their whole jump) | **13.29 → 10.21 ms/forward, −23%**, 3/3 pairs; greedy answers unchanged across 6 prompts (wording-level divergence only) — LOSSY, default ON, app Settings toggle | `--no-decode-attn-quant` / `MLX_SERVE_DECODE_ATTN_QUANT=0` |
> | `fusedQkNormRope` — per-head RMSNorm + RoPE (+ YaRN mscale) for q AND k in one dispatch, cos/sin extracted from a stock-rope probe row (mlxfast 9e06de6, their largest bit-exact win) | **8.94 → 8.80 ms/tok live, −1.6%**, 2/2 pairs both orders — bit-identical, parity-pinned; the eval-per-step ubench read it +1.2% SLOWER (live is the bar) | `MLX_SERVE_QK_NORM_ROPE_FUSED=0` |
> | `MLX_SERVE_WIRED` — zero-headroom residency capacity (mlxfast notes/47: full-wire −28% prefill on THEIR box, and our historical wire-at-max_rec is their documented-bad shape) | **NULL on this M4 Max** — max / fit / off / MLX_MAX_{MB,OPS}_PER_BUFFER=200 all equal within noise, decode AND prefill. Policy implemented, default unchanged (`max`) | `MLX_SERVE_WIRED=fit\|off` |
>
> Mined and settled, do not redo: their decode async ladder (+9.7% for THEM) remains n/a
> here — generate.zig already overlaps at the step boundary; their prefill router top-8
> ranked −0.68% (the 512-row stock sort amortizes — nothing to save); barrier removal
> only pays in 32-thread kernels (their 512-thread variant was ranked-rejected); their
> NVFP4 scale fold is our existing 2^22 fold; their fused router cast/norm sinks are
> inside our `moeRouterTopK` already.
>
> ## Next round, in order (each has a worked reference in the mlxfast tree)
>
> 1. **Certified lm_head prune** — unchanged as the top open lever (~0.9 ms of the 8.8 ms
>    token on Laguna XS, helps EVERY model), and now with a COMPLETE reference
>    implementation to port: `mlxfast-challenge/Sources/MLXFastModel/LagunaLmHeadPrune.swift`
>    (~450 lines). Design: init-time MXFP8 g32 coarse copy (half the bytes), one fused
>    coarse GEMV emitting per-row coarse logit + CERTIFIED bound + bf16 prefill of the
>    output row; a mask kernel (no host readback, no atomics) marking rows whose
>    coarse+bound reaches the max; an exact pass whose per-row arithmetic textually
>    replicates stock `gemv_al_bfloat16` so candidates are bit-identical. Their e4m3/e8m0
>    bit-decoders and the half-ulp bound table are in the file header. Argmax provably
>    stock; gate on `logprobs>0`/grammar → dense fallback (our doc's original plan stands).
> 2. **MoE down-path mega-fusion** — their `laguna_routed_shared_nvfp4_down_residual_bf16`
>    does all 8 routed down-QMVs + shared down + router weighting + fixed-order reduction
>    + the 2.5 scale + BOTH residual adds in one 288-thread dispatch (9 simdgroups, 4
>    output rows per simd — they tuned 1-vs-4 rows/simd TODAY, 4 won). Extends our
>    `gatherQmvGateUp`; same silu-MoE eligibility family. Also graft their
>    residual+RMS+ROUTER fold (`DARKBLOOM_FUSED_RESIDUAL_RMS_ROUTER`): our
>    `fusedAddRmsNorm` null was measured WITHOUT absorbing the router GEMV — folding a
>    real [256, 2048] projection in is the "removes real work" their version has.
> 3. **QK-norm+RoPE fusion for the other QK-norm archs** — wire `gatedFullAttnWith`
>    (qwen3/3.5/3.6 attention layers, hd 128 — the shipped kernel fits as-is) and
>    `gemma4MoeAttnWith` (hd 256 needs a new lane mapping), per-arch A/B each.
> 4. **NVFP4-g16 tail variant of `--decode-attn-quant`** — mlxfast quantizes layers ≥32
>    of 40 to REAL nvfp4 (late-layer amplification ~15x lower); worth another ~6% of the
>    token if quality holds. Same toggle, same side-copy plumbing (`buildAttnDqCopy`
>    grows a mode arg). Re-run the greedy characterization before shipping.
> 5. **Prefill**: their sorted MoE tail (skip `scatterUnsort`'s full expert-bank copy via
>    the inverse permutation) and the prefill fused gate+up bank (+4% prefill on a
>    RUNSKIP-era re-measure — their older "hurts prefill" finding was overturned, ours
>    may be too since it shares provenance with `MLX_SERVE_MOE_GATEUP_FUSED`'s decode-only
>    scoping).
>
> Everything below this line predates the third round; the second-round table and lever 1
> remain accurate history, lever 2 is superseded by next-round item 2 above.

> **Update 2026-07-29 (second round).** Two levers below this line were superseded by a
> round that shipped **−2.7% on Laguna XS decode, byte-identical**, and by one measured
> null result. Read `docs/gotchas/engine-mlx.md` -> "Fusing decode dispatches: only the
> critical path pays" first; the short version:
>
> | change | result | switch |
> |---|---|---|
> | `moeRouterTopK` — the 11-op routing chain in ONE kernel, every MoE arch | **−1.6%**, 3/3 pairs | `MLX_SERVE_MOE_ROUTER_FUSED=0` |
> | `fusedSwiGLU` — `silu(gate)*up` in one dispatch, every silu model | **−1.4%** on MoE, ~0 on dense | `MLX_SERVE_SWIGLU_FUSED=0` |
> | `fusedAttnGate` — Laguna's 4-dispatch output gate in one kernel | **NULL**, ships opt-in | `MLX_SERVE_ATTN_GATE_FUSED=1` |
> | `gatherQmvGateUp` — mlxfast item 3: gate+up+SwiGLU in ONE simdgroup, now the decode default for silu MoE | **−2%** more (whole round **13.338 → 12.783**, −4.2%) | `MLX_SERVE_MOE_GATEUP_FUSED=0` / `MLX_SERVE_MOE_GATHER_DECODE=1` |
> | `fusedAddRmsNorm` — mlxfast item 5's norm tail: residual add + RMSNorm in one kernel | bit-identical but **NULL** (~1% negative), ships opt-in | `MLX_SERVE_ADD_RMSNORM_FUSED=1` |
> | `buildFusedQkv` — mlxfast item 1: q/k/v concatenated into one projection | **NULL at decode AND prefill**, ships opt-in | `MLX_SERVE_FUSED_QKV=1` |
>
> The governing finding, in two halves. **`MLX_SERVE_DISPATCH_PROBE=N` prices a GPU
> dispatch at ~1.5–1.7 µs, but that is an UPPER bound.** A fusion collects it only if
> (a) the launches it removes are not already overlapped — the attention gate depends
> solely on the layer input, so the GPU was already hiding it behind q/k/v and SDPA — and
> (b) it removes real work or a real serialization, not just a launch around an op MLX
> already does well. The residual+RMSNorm pair fails (b): it is genuinely serial, but the
> fused kernel still does 2 reads + 2 writes of a 4 KB row because the residual sum has to
> be emitted, so it trades MLX's tuned `rms_single_row` for one saved 4 KB read.
> The three that paid all replaced something substantial: a 256-wide sort, a three-op
> activation chain, and two whole GEMV passes.
>
> Also settled this round, so do not redo it: `mlx_compile` on the silu geglu is one line
> and NOT output-preserving; a JIT `metal_kernel` and MLX's metallib disagree on `exp` (one
> bf16 value in 65536, hit within ~55 greedy tokens), which is why the SwiGLU sigmoid is a
> table; and the reason `gatherQmv` looked like a losing kernel was ~60% CPU-side config
> construction plus three serial dispatches, both of which item 3 fixed.

## Where things stand (revised)

Laguna XS 2.1 NVFP4, M4 Max, `tests/fwd_ubench.sh`, 20 decode-width forwards per boot:

| | ms/forward |
|---|---|
| tree at the end of 26.7.12 | **13.338** |
| tree after this round | **12.783** (−4.2%) |

Four pairs, alternating which arm boots first, medians: base 13.331 / 13.348 / 13.204 /
13.344 against new 12.823 / 12.787 / 12.762 / 12.778. All four favour the new tree and the
spread inside each arm is under 0.5%.

**This supersedes an earlier −5.8% figure in this document's history**, which chained a
baseline median from one measurement window to a post-change median from another. Same
trap as measurement rule 1 below, committed while writing up the round that discovered it:
a cumulative number assembled from separate windows is not a measurement. The per-lever
numbers below were each taken paired inside one window and stand.

Layer-cap refit on the fixed tree (N = 10/20/30/40): **0.280 ms/layer + 1.55 ms fixed**,
of which ~0.72 ms is lm_head and the rest is largely the probe's own per-forward
`mlx_array_eval` sync, which a real decode loop pipelines away.

### Still open, in the order I would take them

1. **All five mlxfast items are now implemented and measured** — three shipped default-on,
   three measured null and shipped opt-in (item 1's QKV fusion, item 4's attention gate,
   item 5's norm tail). Do not re-attempt the nulls without a new argument; each has its
   numbers and its reason in `docs/gotchas/engine-mlx.md`. Item 1 in particular was the
   doc's own prefill hypothesis and did NOT pay: at M≈7900 each projection is already a
   compute-saturating GEMM, so merging them hides behind the math.
2. **The MoE expert path still has room, even after item 3.** Laguna XS reads ~630 MB
   of expert weights per token and does it at roughly **175 GB/s**, against **412 GB/s**
   for the dense attention projections in the same forward. Closing that gap is worth
   ~2 ms if it could be closed entirely. Item 3 (gate+up+SwiGLU in one simdgroup) took the
   first ~2% of it by spending spare occupancy on instruction-level parallelism, which
   confirms the diagnosis — the kernel is neither ALU- nor bandwidth-bound at those rates
   (~15% of ALU peak, ~32% of memory peak), it is latency-bound. The same lever is not
   exhausted: `down_proj` is still a separate gather, the nvfp4 nibble decode is still
   ~5 ALU ops per value (a bit-parallel decode like `laguna_e4m3_decode4` would cut it),
   and two output rows per simdgroup would double ILP again.
3. **Lever 1 below (certified lm_head prune) still stands**, unchanged, at ~0.72 ms of a
   13.06 ms forward. Note the dense-bf16-head assumption: Laguna XS's `lm_head` really is
   bf16 [100352, 2048] = 411 MB, but dense Qwen3.6-27B's is 4-bit [248320, 640] u32 +
   scales/biases = ~715 MB/token, so a coarse copy for THAT model has to be narrower than
   4-bit and the candidate set will be much larger. Size it per checkpoint.
4. **Lever 2 below (fused routed `[gate; up]` bank) is SUPERSEDED by item 3 above** — The dispatch
   half of its estimate is worth ~0.5% by the probe slope; the streaming half is the
   unknown and is the same thing item 1 attacks. Do it as part of item 1, not separately.
5. **The per-call `metal_kernel` config tax is NOT only in `gatherQmv`.** Measured at
   ~3.3 us of CPU per config build (0.4 ms per 120 calls). Still uncached, found by
   grepping `mlx_fast_metal_kernel_config_new()` in `src/transformer.zig`:
   `verifyQmm` (:826), `runVerifyQmmMsg` (:918), `runVerifyQmmNax` (:1398) and the GDN
   blocked-prefill kernel (in `gatedDeltaNet`). The verify lanes are the interesting ones —
   they fire per projection per layer inside a spec-verify round, so a 40-layer model at
   ~7 projections is ~280 builds ≈ 0.9 ms of pure host time per round against a ~60-80 ms
   round (~1.2%). Left alone deliberately: it could not be measured in this session's
   window and spec-decode cells are the most variance-prone thing in the repo, so it wants
   its own paired run. The fix is mechanical — copy the `GqmvCfgKey` pattern.

5. **Lever 3 below (sliding-layer KV allocation) is unchanged** — still context, not speed.

---

## Lever 1 — certified lm_head prune (firmest estimate, helps EVERY model)

**Estimated gain: ~3% on Laguna XS, ~2.5% on dense Qwen3.6-27B.** Firm, because it is
computed directly from measured lm_head time (0.871 ms of 13.447) and a byte halving.

This is the only lever that pays on a model already sitting at the bandwidth ceiling —
dense Qwen3.6-27B runs at ~427 GB/s of a ~546 GB/s peak, so it has no headroom left except
*reading fewer bytes*.

### Approach

Two-pass: a cheap first pass that bounds each row's logit, then an exact second pass on
only the rows that could still be in the top-k. For a **certified** prune the first-pass
bound must be sound (a true upper bound), or argmax is no longer bit-identical.

For quantized lm_head the natural bound comes from the per-group scales: `|w·x| <=
sum_g |scale_g| * max|x_g| * (max quantized magnitude)`. Compute per row once per token,
which is cheap relative to reading the full packed weight.

### Gating (decided, do not re-litigate)

A pruned projection yields an exact top-k, **not a full logit vector**. So:

- `logprobs > 0`, grammar-constrained decoding, and any full-logit read fall back to
  `lmHeadProject`'s dense path.
- Greedy and top-k/top-p accelerate.

This mirrors the existing `logprobs>0 + grammar disable spec` precedent: one predicate, one
one-shot engagement log, one kill switch. No request gets slower; some get faster.

### Acceptance

- Greedy output **byte-identical** on/off over >=160 tokens, temp 0, on at least Laguna XS
  and a dense Qwen.
- A test that the bound is never violated on random inputs (the prune is only correct if
  the bound is sound — test the bound, not just the output).
- Kill switch + engagement log; a declined prune must be nameable from the log, because a
  rejected guard is output-identical to the fallback and otherwise reads as a null result.

### Files

`src/transformer.zig`: `lmHeadProject` (~:5134) is the single chokepoint — every forward
path goes through it, so there is one place to change. `quantParamsHinted` gives the
per-weight (bits, group_size, mode) you need for the bound.

---

## Lever 2 — fused routed `[gate; up]` expert bank (softest estimate)

**Estimated gain: 2–4% of the token. SOFT — size it properly first.**

Today `MoeMlpWeights` (`src/transformer.zig:3346`) holds `switch_gate_w` and `switch_up_w`
as separate banks, so decode does two gathers and reads `x` twice. Concatenating them at
load into one `[E, 2*inter, hidden]` bank gives one gather and one read of `x`.

Note this must work against **stock `gather_qmm`**, which is now the default on every arch —
not against `gatherQmv`, which was demoted this round.

### Do this first

Get an honest MoE number. Two options, in order of preference:

1. Fix `diagProjBench` to take a varying input. The revert note above says why the obvious
   fix SIGBUSes: keep the f32 source arrays alive for the whole function (the single-input
   form does this by accident), and expect the rungs to get much slower because varying
   inputs defeat MLX's graph cache — raise any probe timeout accordingly.
2. Or bill the real forward directly. Do **not** use `MLX_SERVE_DECODE_PROFILE` for this:
   it forces an eval at every phase boundary, which destroys pipelining and reads 47.4
   ms/token against a real 13.1, with the router phase absorbing sync cost and measuring
   *larger* than the experts it gates.

### Costs to weigh

Touches weight loading and roughly doubles peak load memory for the expert banks during the
concatenation. Check `applyMlxCacheLimit` headroom on a 128 GB machine with Laguna S
(62 GB) before assuming it fits.

### Risk

Structurally low for other models — but the concatenation is at LOAD, so it is not
laguna-gated the way the decode kernel choice is. A/B every arch that has routed experts
(gemma4-26B-A4B, qwen3_5_moe, hy_v3) on its own model before shipping, and keep a kill
switch that falls back to the split banks.

---

## Lever 3 — sliding-layer KV allocation (not speed: CONTEXT)

**Gain: advertised context 163,840 → 262,144 on Laguna XS (+60%). No decode speedup.**

Laguna XS has 10 full-attention layers and 30 sliding layers capped at a 512-token window.
`updateDense` (`src/transformer.zig:~2504`) grows the KV buffer to `entry.offset + new_len`
**regardless of `max_seq`** — the sliding-window cap only narrows the returned VIEW, never
the allocation. So the 30 sliding layers each allocate full-context KV and read 512 tokens
of it.

`computeMemoryContext` (`src/server.zig:2460`) bills all layers at full context, which is
therefore *correct today* — I checked this before trying to "fix" it, and the check is what
stopped a wrong change. Laguna XS lands at 163,840 (memory-bound) instead of its
checkpoint max 262,144 purely because of this over-allocation.

Capping sliding-layer buffers at their window would free ~4x the KV on Laguna and let the
memory model bill honestly. **Both sides must move together** — cap the allocation AND
teach `computeMemoryContext` the per-layer-type model, or auto-context will oversubscribe
into an uncatchable Metal OOM.

### Landmines

- Sliding layers keep the full buffer *by design today* — that invariant is written into
  `CLAUDE.md` and relied on by snapshot/restore, the prefix cache, and the SSD KV tier.
  Changing it means auditing `truncate`/`snapshot`/`restore`/`kv_disk_cache.zig`, all of
  which are currently capacity-agnostic.
- `pinAutoContext` freezes the advertised value at load and clients bake it into config
  files once. Any change here changes what agent CLIs budget against.

---

## Measurement rules for whoever picks this up

These are not general advice; each one cost real time this session.

1. **Never call a regression, or a win, from one run.** Single runs lied three times in one
   session in both directions: a "clean re-run" of gemma4-31b produced high outliers that
   got merged into the baseline, making the next run read as −14%; two repeats put it back.
   `docs/perf-csvs/all-26.7.12.csv` is per-cell **medians** — diff against that, and only
   against same-methodology CSVs.
2. **Attribute before believing.** Every flagged cell this round was structurally
   unreachable from the change under test. Checking reachability is faster than another
   bench run and turns the repeat into a confirmation instead of a hope.
3. **Large models need a settle window between boots.** An 8.4 GB checkpoint's first boot
   after another large load is not a measurement — it read 2x SLOWER, then 5x slower, then
   correctly 15% FASTER once a 45 s cooldown was inserted between paired boots.
4. **Rebuild ReleaseFast before every live A/B.** `zig build test` does not refresh
   `zig-out/bin/mlx-serve`, and Debug fabricates regressions.
5. **Quiet the machine.** Safari/WebKit holding the GPU depressed the two big dense models
   by 25% in one bench run. Check `WindowServer` / `WebKit.GPU` CPU before a window.
6. **A tuning matrix must contain the WIN case, not just the loss cases.** Sweeping the PLD
   yield gate over loss cases alone drove the warmup toward zero, which would have thrown
   away a +77% win that only shows on echo workloads.
7. **Position-balance the pairs.** Running the control first in every pair let a slow
   upward drift land entirely on one arm: the same six boots read −1.9% in one order and
   −2.8% in the other, medians −2.7%. Alternate which arm goes first.
8. **Sweep a 16-bit domain, don't sample it.** Random inputs said a fused activation was
   bit-identical; the live model diverged at token ~55 on one of the 65536 bf16 patterns.
   Any kernel whose input is a 16-bit float has a finite, cheap, complete test.
