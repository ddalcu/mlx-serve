# KV-quant performance plan (4-bit / 8-bit / TurboQuant)

Status: **IMPLEMENTED 2026-07-30** (Phases 0–2 + a decode kernel that replaced Phase 1's
composed path as the main lever; Phase 3 SKIPPED with reason, Phase 4 partially superseded —
see "Outcome" below). Audience: a future agent picking this up cold.

## Outcome (2026-07-30)

- **The fused flag was a silent no-op on the live decode path** before this round:
  `updateAffine` (what every decode forward calls) never set the quant triples on its
  returned `DenseKVView` — only the read-only `denseView` did. `--kv-attn-mode fused`
  engaged on nothing; tests/measure scripts compared dense against dense. Fixed + engagement
  now asserted in `tests/test_kv_quant_fused_equivalence.sh` (fused arm must log both
  `[kv-attn] fused engaged` and `[kv-attn] decode kernel engaged`; dense arm neither).
- **Phase 1 (grouped-Q composed) shipped but is NOT the perf lever**: the Phase 0 µbench
  showed composed qmm at M=g rows over batched 4-D banks runs ~6x below bandwidth and LOSES
  to dense-mode reads at every real GQA ratio (48/8 @32K: composed 1.47 ms vs dense 1.11 vs
  off 0.51). It remains as the verify-width (T_q 2..32) fallback.
- **The lever is `qkv_attn_dec`** — a custom metal_kernel reading the packed affine K/V
  views IN PLACE at decode width (T_q=1): grid over (kv_head × kv_block), per-block scores
  computed row-per-thread with no reduction on the critical path, thread-per-element V
  accumulate reading each packed row ONCE regardless of GQA ratio, per-block (m, l, O)
  partials merged by a handful of KB-scale mlx ops. Threadgroup footprint kept ≤ ~10 KiB —
  the first two designs measured 5x/1.7x SLOWER from occupancy starvation (48 threadgroups /
  28 KiB tg-memory respectively); block size and tg-memory are ONE decision.
- **Live same-boot interleaved A/B** (per-request `kv_attn_mode` field, prefix-cache-served
  long prompt so each request measures decode, server's own `timings`):

  | Model | ctx | kv off | kvq8 dense | kvq8 fused | fused vs dense |
  |---|---|---|---|---|---|
  | Laguna XS NVFP4 (48/8, hd128) | 10.7K | 96.0 | 70.3 | 77.5 | **+10%** |
  | Laguna XS NVFP4 | 42K | 70.0 | 37.0 | 57.7 | **+56%** |
  | Qwen3.6-27B 4b (hybrid) | 37K | · | 17.0 | 21.4 | **+26%** |

  (tok/s, temp 0, 3/3 resp. 2/2 pairs each, answers same-prefix.) kv-quant remains a
  memory feature — fused does not beat kv-off on Laguna (~-18% at 42K vs off) — but the
  long-context penalty drops from −47% to −18%.
- **Phase 2 shipped as default**: `--kv-attn-mode auto` (new default) engages fused reads
  when scheme==.affine and prompt ≥ 8192 tokens (`server.KV_ATTN_AUTO_CROSSOVER_TOKENS`);
  a per-layer kv floor (`KV_ATTN_FUSED_MIN_TK` = 1024, env-overridable
  `MLX_SERVE_KV_ATTN_MIN_TK`) keeps short-KV layers (Laguna's 512-token sliding windows —
  measured −2% when they fused: the split-KV merge ops don't pay at tiny T_k) on dense
  reads even inside a fused request. Per-request `kv_attn_mode: "dense"|"fused"` outranks
  everything. Kill switches: `MLX_SERVE_KV_ATTN_FUSED=0` (whole route),
  `MLX_SERVE_KV_ATTN_KERNEL=0` (kernel only → composed).
- **Wired sites**: `forwardStandardWith`, `gatedFullAttnWith`, `lagunaAttnWith` (the A/B
  read dense-vs-dense until laguna was wired — engagement counting caught it, again).
  `gemma4MoeAttnWith` (26B-A4B) deliberately NOT wired yet — kv-quant there silently keeps
  dense reads (honest, just not accelerated). `forwardBatchedDecode`/`diffusionDecoderAttn`
  keep dense reads by design.
- **Two latent bugs this round flushed out**: (1) the composed causal arm built its mask as
  `upper * -inf` — `0 × -inf = NaN` poisoned every sub-diagonal entry, live symptom gemma
  answering `<pad><pad>`; (2) every parity loop was NaN-BLIND (`NaN > max_err` is false),
  which is how (1) shipped green. All parity loops now assert finiteness first; the mask is
  built with `mlx_where`.
- **Phase 3 (TurboQuant fused) SKIPPED**: it would ride the composed path, which loses at
  GQA ratios; the real fix is teaching the kernel rotation-aware dequant — separate session.
- **Phase 4 (prefill flash kernel)**: NOT built; the decode kernel covers where kv-quant
  lives or dies (the doc's own call). Prefill under kv-quant still pays dense
  materialization + eval-per-layer cadence.

The original plan follows for context; read the `## Rules` engine section in the root
CLAUDE.md before touching any of this — every trap named there has already been hit once.

## Goal

`--kv-quant 4|8|turbo2|turbo4` currently trades decode/prefill SPEED for KV memory. The goal
is to make quantized KV a net performance WIN at long context (where its memory savings
matter most) and at worst neutral at short context, without changing the quality contract
(kv-quant divergence thresholds per arch stay as they are).

## Current architecture (verified 2026-07-30)

One cache type, `transformer.KVCache`, scheme-dispatched:

| Piece | Where | What it does today |
|---|---|---|
| `KVCache.update` | src/transformer.zig ~2361 | dispatches `.off` → `updateDense`, `.affine` → `updateAffine`, `.turboquant_*` → `updateTurboQuant` |
| `updateAffine` | ~2420 | quantizes incoming K/V (`mlx_quantize`, affine, group_size 64), 6 buffers (q/scales/biases × K,V), grow + `slice_update` writes, builds 6 sliced views, then **dequantizes the full views to dense bf16** and returns them as the `DenseKVView` |
| `updateTurboQuant` | ~2372 | rotates K and V by per-layer Hadamard matrices (`kv_quant.TurboState`, lazy-built), then rides `updateAffineRotated` (same 6-buffer machinery) |
| `KVCache.denseView` | ~2629 | read-without-write path: `.off` returns raw views; `.affine` dequantizes full K/V AND carries the packed triple alongside (`has_quant_triple`, `k_triple_*`/`v_triple_*`); `.turboquant_*` dequantizes + un-rotates, NO triple |
| `kv_quant.quantAttention` | src/kv_quant.zig ~479 | "Plan ricky" fused path: composed attention reading the packed triples directly via `mlx_quantized_matmul` (scores: `transpose_w=true` contracting D; out: `transpose_w=false` contracting T_k), precise softmax, causal/array/no mask |
| `ForwardCtx.kv_attn_fused` | transformer.zig ~3826 | opt-in gate for quantAttention. Wired at TWO attention sites: `forwardStandardWith`'s attention and `gatedFullAttnWith`. `.affine` only — TurboQuant and `.off` ignore the flag |
| `--kv-attn-mode dense|fused` | main.zig ~704, server.zig `default_kv_attn_fused` | server default for new requests; a per-request `kv_attn_mode` body field overrides. Default: dense |
| `prefillEvalCadence` | transformer.zig ~6346 | under kv-quant forces eval-per-LAYER during prefill (the dequant transient is GB-scale: 2 × 600K × 8 × 128 × 2B ≈ 2.46 GB — pinned by the `"quantized-KV dequant forces eval-per-layer"` test) |
| Sliding-window slice | `updateAffine` step 7 | decode views slice the 6 PACKED buffers to the last `max_seq` BEFORE dequant — slice-before-dequant is already right on this path |
| Snapshots | KVCache snapshot/restore | 6 handles in quant mode; restore is offset-truncate, capacity-agnostic |
| Batched decode | `forwardBatchedDecode` ~7510 | reads `denseView` (dense materialization), NO fused arm — the packed-words-to-SDPA server-kill class lives here, guarded by the force-batched × kv-quant section of `tests/test_batched_equivalence.sh` |
| Diffusion | `diffusionDecoderAttn` ~8157 | reads `denseView`, no fused arm |

Parity tests that already exist (src/kv_quant.zig bottom): `quantAttention` vs dense SDPA at
4-bit decode (T_q=1) and causal prefill (T_q=T_k=4), plus quantized-matmul transpose-mode
probes. They are correctness tests only; nothing pins engagement or perf.

## Cost model — why each mode robs performance today

Let N = dense KV bytes for the current context (2 × heads_kv × T × head_dim × 2B per layer).

**Decode, dense mode (the default, what users get):** every layer, every token:
read packed (N/2 at 8-bit, N/4 at 4-bit) + WRITE dense N + SDPA reads dense N.
Total ≈ 2.25–2.5 × N vs. exactly N with kv-quant off. KV traffic dominates long-context
decode (measured 2026-07-29: the KV term at 32K on Laguna XS is ~4.4 ms/token at 307 GB/s —
see the "3x forward fix" rule), so kv-quant decode is strictly SLOWER than dense KV today.
The memory win is real; the speed is negative.

**Decode, fused mode (opt-in, `--kv-attn-mode fused`):** reads packed directly — the right
shape — BUT `quantAttention` handles GQA by `mlx_repeat_axis`-ing all six triple components
from H_kv to H_q (kv_quant.zig ~529). The comment claims stride-0 views defer the copy, but
`mlx_quantized_matmul` makes non-contiguous operands contiguous on read: if that fires, every
layer/token materializes the packed bank ×(H_q/H_kv) — ×6 on Laguna XS (48/8), ×4 on Qwen
3.5/3.6 — which can exceed the dense read it was saving. UNVERIFIED SUSPICION: measure first
(Phase 0). It also loses flash-attention tile fusion (scores materialize in HBM), which is
fine at decode widths and fatal at prefill widths.

**Prefill, any quant mode:** `denseView` rebuilds the FULL dense cache per layer per chunk →
GB-scale transient → `prefillEvalCadence` drops to eval-per-layer → pipeline drains. Two
separate costs: the dequant bandwidth AND the lost pipelining. This is why long-context
prefill under kv-quant feels disproportionately slow.

**TurboQuant additionally:** dequant + un-rotate on every read, and it is excluded from the
fused path entirely, so it always pays the dense-materialization tax plus a [D,D] matmul per
read.

**Write path (minor):** per decode token, `updateAffine` runs quantize + 6 grow-checks + 6
slice_updates + 6 view builds — CPU graph-build ops, not GPU-bound. Only worth touching after
the big levers.

## What is already known (do NOT re-derive)

- Slice-before-dequant on sliding windows is already correct in `updateAffine` (packed views
  sliced to `max_seq` first). Verify the same holds on the `denseView` read-only path before
  assuming.
- `mlx-c` (pinned fba4470 ≈ 0.6.0) exposes NO native quantized SDPA — `grep quantized_scaled
  lib/mlxc-src/mlx/c/*.h` comes back empty. Any packed-reading attention is ours: composed
  qmm (exists) or a custom `metal_kernel` (doesn't yet).
- `mlx_quantized_matmul` IS the right primitive for dequant-free reads (same one `qmatmulBits`
  uses for weights); `gatherQmv` proves we can hand-roll affine AND nvfp4 dequant inside our
  own Metal kernels when the stock op doesn't fit.
- INT4 long-greedy divergence is legit and documented (qmv-vs-qmm reduction order); kv-quant
  divergence stacks on top with per-arch first-N thresholds. The fused path does not get to
  make this WORSE silently — parity tests compare against the dense-mode kv-quant output, not
  against kv-quant off.
- The hot prefix cache records its scheme per entry (`HotEntry`) — no cross-scheme hits. The
  disk tier stores what the hot tier holds. Neither needs to change for any phase below.

## The plan

### Phase 0 — instrument and baseline (no behavior change)

1. One-shot engagement logs for the fused path (`[kv-attn] fused engaged: bits=.. gs=.. Hq/Hkv=..`),
   mirroring every other kernel. Engagement COUNTS in tests, never output equality alone —
   the two shipped `use_drafter=false` call sites are the cautionary tale.
2. A µbench answering the GQA-repeat question directly: time `quantAttention` decode
   (T_q=1) at H_q/H_kv ∈ {1, 4, 6}, T ∈ {2K, 8K, 32K}, bits ∈ {4, 8} vs. (a) dense-mode
   denseView+SDPA and (b) kv-quant off. Give every layer its own buffers (the 449 GB/s
   working-set trap). µbench wins can lose live — this only SCOPES the work, the live A/B
   decides defaults.
3. Baseline live numbers with `tests/bench.sh` single-rung methodology (fast-iteration rule:
   ONE rung, mlx-serve only) on: Laguna XS (hd 128, 6:1 GQA, MoE), Qwen3.6-27B (hybrid — only
   full-attn layers have KV), gemma-4-26B-A4B or 31b (hd 256, sliding windows). Three cells
   per model: kv-quant off / 8 dense-mode / 8 fused-mode, at 8K and 32K ctx.

Deliverable: a table in this file replacing the UNVERIFIED SUSPICION above with numbers.

### Phase 1 — make the fused decode path actually fast (the big lever)

1. **Kill the GQA repeat.** Reshape Q instead of expanding K/V: `[B, H_q, T_q, D]` →
   `[B, H_kv, g·T_q, D]` where `g = H_q/H_kv` (view: split H_q into [H_kv, g], merge [g, T_q]
   into rows). The packed triples are then read IN PLACE by `mlx_quantized_matmul` with no
   repeat at all. Output `[B, H_kv, g·T_q, T_k]` → attn @ V → `[B, H_kv, g·T_q, D]` →
   reshape back. Masking under grouped rows: row r corresponds to token position r % T_q, so
   the causal mask is the `[T_q, T_k]` mask TILED g times along rows (`mlx_tile` or build at
   `[g·T_q, T_k]` directly). T_q == 1 (pure decode) needs no mask at all — start there.
2. **Cover the verify widths.** PLD/drafter/MTP verify forwards are T_q 2..9 and pay the same
   full-cache dequant per layer; with the tiled mask they ride the same fused path. Spec
   invariants unchanged (`cache.step`, t1-not-in-cache, correction sampling).
3. **Explicit declines, never silent ones.** `forwardBatchedDecode` and `diffusionDecoderAttn`
   keep dense-mode reads in v1 — write the decline into the predicate with a comment, and keep
   the force-batched × kv-quant section of `tests/test_batched_equivalence.sh` green. A new
   eligibility predicate (`kvAttnFusedEligible`: scheme==.affine, has_triple, T_q ≤ 32,
   supported mask mode) is the ONE gate both call sites read.
4. **TDD:** extend the kv_quant.zig parity tests to grouped-Q shapes (H_q≠H_kv at T_q ∈
   {1, 4, 9}, bits ∈ {4, 8}, gs 64) against dense SDPA over dequantized K/V — no-worse-than
   reference, never kernel-vs-kernel. Red first by asserting the repeat path is GONE (source
   scan for `mlx_repeat_axis` in quantAttention, or better: an op-count/engagement assert).
5. **Live acceptance:** same-boot interleaved A/B (the per-request `kv_attn_mode` body field
   is the interleave lever — same trick as the lm-head prune's `presence_penalty: 1e-9`).
   Fused must beat dense-mode decode at 8K and 32K on Laguna XS and the Qwen; record where it
   crosses kv-quant-off decode. Quality: 6-prompt greedy characterization per arch, answers
   within the existing kv-quant divergence expectations.

Kill switch: `--kv-attn-mode dense` stays, plus `MLX_SERVE_KV_ATTN_FUSED=0` env for A/Bs.

### Phase 2 — auto mode (make the win reachable without flags)

`--kv-attn-mode auto` as the NEW DEFAULT once Phase 1 numbers are in: per request, pick
fused when scheme==.affine and effective context ≥ the measured crossover (constant per
arch family is fine to start; store it next to the other calibrated constants, not hardcoded
at call sites). Short-context requests keep flash SDPA on the dense view. The explicit
per-request `kv_attn_mode` keeps outranking auto. App side: nothing — ServerOptions doesn't
expose kv-attn-mode, and auto means it never needs to. Update `--help`, README flags table,
and the root CLAUDE.md flag list (flag cross-check rule: main.zig match list is the source of
truth the app and tests are checked against).

### Phase 3 — TurboQuant joins the fused path

The Hadamard rotation is orthogonal, so stop un-rotating the cache:
- K: `scores = (R_k q) · (R_k k)` — rotate Q once per layer ([1,H,T_q,D] @ [D,D], tiny) and
  read packed rotated K directly.
- V: `out = R_v^T · (softmax(scores) · (R_v v))` — read packed rotated V directly, un-rotate
  the OUTPUT once per layer (another tiny [D,D] matmul).
This removes TurboQuant's full-cache dequant+unrotate per read entirely. `denseView` keeps
the dense fallback for non-eligible paths. Extend the triple plumbing (`has_quant_triple`)
to the turbo arms with the rotation handles alongside. Parity: fused-turbo vs dense-turbo
(current behavior) — bit equality is NOT expected (different contraction order); use the
no-worse-than-reference bar with the dense fp32 dequant ground truth, same as every kernel
test. Pow-2 head_dim precondition already holds (TurboQuant requires it).

### Phase 4 — prefill: in-kernel dequant flash attention (the hard one)

The composed fused path can never serve prefill (scores `[H, chunk, total_kv]` materialize —
at 2048 × 600K that is the exact thing flash attention exists to avoid). The fix is a custom
`metal_kernel` flash-prefill that dequantizes K/V TILES in-kernel:

- Start from `msv_attn_p256`'s structure (tiled flash with m/l/O carry, band + causal arms,
  BK-aligned kv chunking) and `gatherQmv`'s affine dequant codegen (shift-based, no LUT).
- Lanes: hd 128 first (Laguna, Qwen dense-attn layers — MLX's own SDPA owns this dim today,
  so the win is purely skipping the dense KV materialization), hd 256 second (gemma4/qwen3.5
  — fold into the existing p256 kernel as a quantized-K/V variant rather than a new file).
- Sliding-band arm included from day one (gemma local layers are where the GB-scale masks
  and transients hurt most).
- Once prefill attention reads packed KV, `prefillEvalCadence`'s kv-quant arm can relax back
  to the coarse cadence — that is a SECOND, separate win (pipelining back). Change the
  cadence predicate and its pinned test TOGETHER, gated on the kernel actually being
  eligible for every layer of the model (mixed eligibility keeps eval-per-layer).
- Every dtype read off the array, block sizes follow the dtype (the GDN blocked-prefill
  class); signature from actual dtypes, <8-element arrays land in `constant` (the
  metal_kernel signature class); parity tests cover the LIVE dtypes and REAL geometries
  (reduction-order tests need contraction dims near production size — the fused-QKV lesson).
- A/B per arch before default-on (eligibility predicates adopt every matching shape).

This phase is the largest and can ship LAST or not at all if Phase 1+2 already clear the
bar David cares about; decode is where kv-quant lives or dies.

### Phase 5 — write-path and polish (only if profiling says so)

- Fuse quantize+write: one custom kernel quantizing the incoming [1, H_kv, 1, D] token and
  writing q/scales/biases at offset, replacing quantize + 6 slice_updates (~CPU graph-build
  savings, the gatherQmv-config-rebuild class of cost). Config caches keyed with `ShapeKey`
  — never products (the 2026-07-29 server-kill class).
- Group-size experiment: gs 32 at 4-bit for quality headroom vs. the extra scales traffic —
  quality first (long-greedy divergence thresholds), perf second.
- Snapshot cost under quant (6 handles): already offset-truncate + capacity-agnostic; verify
  no accidental copy-on-write pins after the fused path lands (the no-KV-snapshots-across-
  verify rule).

## Test matrix to leave behind (guards, not just green runs)

| Guard | What it pins |
|---|---|
| kv_quant.zig parity (extend) | grouped-Q fused == dense-SDPA-over-dequant, all bits × gs × T_q ∈ {1,4,9} × GQA ratios {1,4,6}, both mask modes |
| Engagement assert | fused path ENGAGES on an eligible request (log-scrape or counter), and DECLINES on batched/diffusion — silent fallback is output-identical, so equality tests can't see it |
| `tests/test_batched_equivalence.sh` | force-batched × kv-quant stays green (the packed-words-to-SDPA kill) |
| Long-greedy quality | per-arch first-N byte-stability under kv-quant unchanged from dense mode (thresholds already exist, env-overridable) |
| Spec × fused | PLD/MTP engagement + equivalence with kv-quant fused on (verify forwards ride the new path) |
| Hot-cache × fused | restore then decode under fused mode (the 16-token restore-suffix class showed restore paths find shape holes nothing else hits) |
| TurboQuant fused parity | vs dense fp32 ground truth, both turbo variants |
| Cadence test pair | `prefillEvalCadence` kv-arm relaxation lands in the SAME change as its pinned test |

## Bench discipline (binding, from /bench + root rules)

- Rebuild `zig build -Doptimize=ReleaseFast` before ANY live number — `zig build test` does
  not refresh the exe (this exact mistake cost an hour on 2026-07-29).
- Same-boot interleaved A/Bs via the per-request `kv_attn_mode` field; alternate per BOOT
  when interleave isn't possible; 45 s settle between boots for big models.
- Mid-iteration checks: ONE rung, ~3 min. Full ladders only for the final verdict, medians
  over repeated runs, diff only against same-methodology CSVs.
- Never quote a win without naming what it is over (kv-quant-off dense KV vs kv-quant dense
  mode are DIFFERENT baselines — say which).
- Attribute before believing any regression in cells the change cannot reach.

## Risks / traps specific to this work

- `mlx_quantized_matmul` batching/contiguity semantics are the load-bearing unknown for
  Phase 1 — verify with the Phase 0 µbench before building on the reshape.
- Composed attention materializes scores `[.., g·T_q, T_k]` — at decode this is KB–MB scale
  (fine); cap eligibility (T_q ≤ 32) so nothing routes a real prefill through it.
- Grouped-Q changes the reduction ORDER vs dense SDPA — expect qmv-class near-tie argmax
  flips at temp 0; that is sanctioned (same class as verifyQmm), pin no-worse-than, not
  bit-equal.
- `denseView`'s laziness means today's dense arrays cost nothing UNLESS read — when wiring
  new call sites, make sure the fused branch actually stops referencing `.k`/`.v`, or the
  dequant graph still executes and the "win" is zero while everything stays green.
- Uncatchable MLX shape errors kill the server: every new packed-read path proves its shapes
  before dispatch (decline + log, never assert), and anything cached is keyed by `ShapeKey`.
- macOS/Metal: threadgroup memory budget for the Phase 4 kernel at fp32 carry + 256-wide
  tiles — clamp block size by dtype like `gdnBlockTFor`.

## Out of scope

- Changing quantization FORMAT (nvfp4 KV, mixed per-layer bits) — separate investigation.
- mlx/mlx-c upstream bumps to chase a native quantized SDPA (none exists in the pinned pair;
  re-check on the next planned bump, and if upstream grows one, Phase 4 shrinks to an FFI
  extern + parity tests).
- Multi-model / LAN interactions — kv-quant is per-cache, nothing crosses the transport.
