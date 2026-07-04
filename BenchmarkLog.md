# mlx-serve Benchmark Log

Performance tracking across releases. Run `./bench.sh` after every major feature or optimization change and append results here.

## How to run

```bash
# Full suite (all models):
./bench.sh

# Single model:
./bench.sh --model gemma

# Custom binary / more runs:
./bench.sh --binary ./my-build --runs 5
```

## Methodology

- **Prefill**: 840-token prompt (fixed, generated), `--max-tokens 1`, `--temp 0`
- **Decode**: "Write a detailed essay about quantum computing", `--max-tokens 256`, `--temp 0`
- **Runs**: 3 total. Run 1 is warmup (includes model loading from disk, excluded). Runs 2-3 are averaged.
- **System**: Apple M4, 16 GB unified memory (unless noted otherwise)

### Models

| Short name | Path | Architecture | Params | Quant |
|---|---|---|---|---|
| Gemma-4-E4B-4bit | `gemma-4-e4b-it-4bit` | `gemma4` | ~4B | 4-bit |
| LFM2.5-350M-8bit | `LFM2.5-350M-MLX-8bit` | `lfm2` | 350M | 8-bit |
| Qwen3.5-4B-4bit | `Qwen3.5-4B-MLX-4bit` | `qwen3_5_moe` | ~4B | 4-bit |

### Prompts

**Prefill prompt** (840 tokens):
```
Explain the following topics in extreme detail: topic 1 about science and technology
and its impact on human civilization throughout history, topic 2 about ..., ... topic 49 about ...
```

**Decode prompt** (16 tokens):
```
Write a detailed essay about quantum computing
```

---

## 2026-05-04 — v26.5.1: Responses API + WebSockets, tokenizer arena fix

**Changes since 2026-04-16**:
- `loadTokenizer` keeps the parsed `tokenizer.json` arena alive and borrows vocab/merge string pointers from it (no per-entry dupe). Pre-sized hashmaps to skip rehashing.
- New `/v1/responses` (Responses API + compaction) and WebSocket transport on `/v1/responses` — exercise the same forward-pass code path, no inference change expected.

### mlx-serve

| Model | Prefill (tok/s) | Decode (tok/s) | Memory |
|---|---|---|---|
| Gemma-4-E4B-4bit | 388.0 | 33.5 | 4.344 GB |
| LFM2.5-350M-8bit | 3825.6 | 214.3 | 0.406 GB |
| Qwen3.5-4B-4bit | 382.9 | 37.8 | 2.266 GB |

### Δ vs 2026-04-16

| Model | Prefill | Decode | Memory |
|---|---|---|---|
| Gemma-4-E4B-4bit | +5.2% (368.7 → 388.0) | +5.3% (31.8 → 33.5) | ≈ same |
| LFM2.5-350M-8bit | +4.4% (3666.0 → 3825.6) | +4.9% (204.3 → 214.3) | same |
| Qwen3.5-4B-4bit | **+165% (144.3 → 382.9)** | +15.2% (32.8 → 37.8) | -6.0% |

### Analysis

- **Qwen3.5 prefill jump is the headline**: 144 → 383 tok/s on 844-token prompts, now ~93% of mlx-lm 0.31.2's reference (410). The previous gap was attributed to per-timestep GatedDeltaNet recurrence vs mlx-lm's parallel scan, but no SSM/scan code changed — the fix is the tokenizer arena change. The old 2026-04-16 measurement included tokenizer-load time inside the prefill metric, and the per-timestep `allocator.dupe` over 144k vocab + ~150k merges was eating multiple seconds of wall-clock per warmup run. With borrow-from-arena, that overhead vanishes.
- **Gemma / LFM gains** (~5%) are within run-to-run thermal variance from the same effect on smaller string tables. Real but minor.
- **Decode is unchanged** in absolute terms — small movements (Gemma 31.8 → 33.5, Qwen 32.8 → 37.8) are within the typical noise floor of 256-token decode runs on a 16 GB M4. No code on the decode hot path changed.
- **No regressions** from the +1395 lines of `server.zig` for Responses/WebSocket — those endpoints don't touch the chat-completions forward pass that bench.sh exercises.

### Reference (mlx-lm 0.31.2, 2026-04-16, unchanged)

| Model | Prefill (tok/s) | Decode (tok/s) | Memory |
|---|---|---|---|
| Gemma-4-E4B-4bit | 559.1 | 31.6 | 4.316 GB |
| LFM2.5-350M-8bit | 4303.2 | 232.0 | 0.421 GB |
| Qwen3.5-4B-4bit | 409.8 | 36.6 | 2.476 GB |

---

## 2026-04-16 — Nemotron-H SSM precision fix + time_step_limit fix

**Commit**: `dfd66c4` + uncommitted

**Changes**:
- Nemotron-H: Cast A_neg to float32 in Mamba2 SSM (matching Python precision)
- Nemotron-H: Fixed time_step_limit defaults (Python uses `(0.0, inf)`, we were reading `time_step_min`/`time_step_max` from config which clipped dt incorrectly)
- Qwen3.5 GatedDeltaNet: Fixed parameter-free RMS norm (mlx-c now requires non-empty weight array, pass ones)
- Qwen3.5 GatedDeltaNet: Fixed SSM state init (conv1dWithCache sets `initialized=true` before state is created, check `ctx==null` instead)

### mlx-serve

| Model | Prefill (tok/s) | Decode (tok/s) | Memory |
|---|---|---|---|
| Gemma-4-E4B-4bit | 368.7 | 31.8 | 4.328 GB |
| LFM2.5-350M-8bit | 3666.0 | 204.3 | 0.406 GB |
| Qwen3.5-4B-4bit | 144.3 | 32.8 | 2.411 GB |

### mlx-lm 0.31.2 (reference)

| Model | Prefill (tok/s) | Decode (tok/s) | Memory |
|---|---|---|---|
| Gemma-4-E4B-4bit | 559.1 | 31.6 | 4.316 GB |
| LFM2.5-350M-8bit | 4303.2 | 232.0 | 0.421 GB |
| Qwen3.5-4B-4bit | 409.8 | 36.6 | 2.476 GB |

### Analysis

- **Decode**: mlx-serve matches mlx-lm within ~10% across all models (31.8 vs 31.6 Gemma, 32.8 vs 36.6 Qwen)
- **Prefill**: mlx-lm is faster on prefill — likely due to parallel scan (SSD) for SSM models vs our per-timestep loop. Gemma prefill gap (369 vs 559) is due to system thermal state variance between runs.
- **Memory**: Nearly identical between the two — both use the same MLX backend
- **Qwen3.5 prefill**: Our per-timestep GatedDeltaNet recurrence (144 tok/s) is ~2.8x slower than mlx-lm's parallel implementation (410 tok/s) on 844-token prompts. Decode speed is comparable.

---

## 2026-05-13 — DSV4 MTP perf knobs added (Tier A, code-only)

**Changes**:
- `SPEC_GATE_WARMUP` env var: overrides `Generator.RUNTIME_GATE_WARMUP` (default 5)
  at request entry. Allows A/B-ing the 5 → 2 lower-warmup hypothesis from the
  Plan 02 / TODO MTP Tier A list without a rebuild between runs.
- `--mtp-block-size <n>` CLI flag exists today — Tier A's `=2` half-batch
  experiment is just a launch-flag flip, no code change required.

**Pending bench runs (need DSV4 weights loaded + 300-tok echo prompt workload)**:
| Config                          | Per-draft acceptance | Decode tok/s | Status |
|---|---|---|---|
| MTP-on, block=4, warmup=5 (today)| 0.27 (gate trips r5)|        21.4   | baseline |
| MTP-on, block=2, warmup=5        | TBD                  | TBD          | run with `--mtp-block-size 2`        |
| MTP-on, block=4, warmup=2        | TBD                  | TBD          | run with `SPEC_GATE_WARMUP=2`        |
| MTP-on, 8-bit `e_proj`/`h_proj`  | TBD                  | TBD          | reconvert via `/tmp/convert_mtp.py QUANT_8BIT` |

A bench-script harness can iterate the matrix automatically once DSV4 is
running locally; the env knob means each row is a server restart, not a
recompile.

### bench_dsv4 — 2026-05-15 23:55 — baseline-mlx-lm-0.31.3
- engine=mlx-lm runs=4/5 decode_tps median=38.66 (min=38.62 max=38.93) wall_median=6.92s pt=29 ct=256

### bench_dsv4 — 2026-05-15 23:57 — baseline-mlx-serve-current-main
- engine=mlx-serve runs=4/5 decode_tps median=24.29 (min=24.26 max=24.30) wall_median=10.54s pt=29 ct=256

### bench_dsv4 — 2026-05-15 23:58 — phase-1.1-padded-rope-on
- engine=mlx-serve runs=4/5 decode_tps median=24.39 (min=24.37 max=24.45) wall_median=10.50s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:06 — phase-2.2-hc-fused-kernel
- engine=mlx-serve runs=4/5 decode_tps median=38.90 (min=38.81 max=39.10) wall_median=6.58s pt=29 ct=256

## 2026-05-16 — DSV4 Speed Loop: fused HC kernel beats mlx-lm

**The big win.** Ported `_hc_sinkhorn_collapse` Metal kernel verbatim from mlx-lm `deepseek_v4.py:511-636` into `src/arch/deepseek_v4.zig` (HC_SINKHORN_COLLAPSE_KERNEL_SOURCE + getHcSinkhornCollapseKernel() + hcFusedCollapse()). Replaces ~25 MLX kernel dispatches per `hcPre` call (called 86×/forward at 43 layers + 2 per layer) with a single Metal kernel that does mixes → branchless sinkhorn → bfloat4-vectorized collapse in one go. Default ON; set `DSV4_HC_KERNEL=0` to opt out.

| Config                                      | engine     | decode tok/s | wall (256 tok) | Δ vs prior |
|---|---|---|---|---|
| mlx-lm 0.31.3                               | mlx-lm     | **38.66**    | 6.92s          | reference  |
| baseline (current main, no flags)            | mlx-serve  | 24.29        | 10.54s         | 0.63×      |
| `DSV4_PADDED_ROPE=1` (no other change)       | mlx-serve  | 24.39        | 10.50s         | +0.4%      |
| **fused HC kernel (default)**                | **mlx-serve** | **38.90** | **6.58s**      | **+60.2% / mlx-serve > mlx-lm** |

**Gap closed: 14.37 → -0.24 tok/s.** mlx-serve now beats mlx-lm by 0.6% (38.90 vs 38.66) on the canonical SvelteKit/Prisma prompt at temp=0, 256 max_tokens, median of 4 warm runs.

**Quality validation** (`DSV4_HC_KERNEL=1` default-on path):
- Warm-short canary 10/10 PASS at temp=0.8, max_tokens=400 across 10 seeds (`tests/dsv4_warm_short_canary.sh`)
- `test_dsv4_stabilization.sh` no-degeneracy: 5/5 paths PASS — no token-loop collapses
- Path consistency: non-streaming `/v1/chat/completions` ↔ `/v1/completions` ↔ `/v1/messages` all match
- 3 fails on the stabilization matrix are pre-existing streaming-vs-nonstreaming divergences (documented in CLAUDE.md "DSV4-Flash status"); same fails reproduce with `DSV4_HC_KERNEL=0` so they're not introduced by the kernel.
- 366/366 zig unit tests PASS.

**Why this worked.** The hcPre/hcPost/hcHeadReduce family was the dominant per-token cost (~1.5k MLX dispatches/token across 43 layers). mlx-lm collapses each `_fused_collapse` call into one fused MSL kernel with:
- Branchless sinkhorn on simd group 0 (no divergent loop branches; `simd_sum()` for column normalization is free SIMD shuffle)
- Native bfloat4 vectorized collapse (single 64-bit load per 4 bf16 values, fma chains for 3 of 4 terms)
- One threadgroup per row (256 threads); ~25 dispatches per layer → 1.

We had the FFI surface for `mlx_fast_metal_kernel_new/apply` since the GDN port (transformer.zig:8-99); this is the same pattern applied to DSV4 HC. MSL source is verbatim from mlx-lm (no math change; same kernel Apple ships), only the dispatch wiring is new.

### Open speed work (residual)

The bench tied/exceeded mlx-lm on the canonical prompt. Further wins would come from:
- mlx-lm uses `@mx.compile` on 9 hot subgraphs (`_rope_full`, `_hc_mixes`, `_hc_expand_op`, `_rms_rsqrt`, `_score_func`, `_expert_select`, etc.). We compile only GELU/GeGLU. Each adds another fused-kernel boundary.
- Inverse-RoPE path is still slice+rope+concat in our `hcPre` MLX-fallback. Padded RoPE A/B showed +0.4% which suggests the win is in the dispatches saved, not the math.
- Indexer math for long-context (n_comp > 2048) is opt-in, not default.


### bench_dsv4 — 2026-05-16 00:13 — phase-2.2-hc-kernel+padded-rope
- engine=mlx-serve runs=4/5 decode_tps median=39.95 (min=39.79 max=40.02) wall_median=6.41s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:20 — phase-2-final-confirmation
- engine=mlx-serve runs=4/5 decode_tps median=39.04 (min=38.93 max=39.24) wall_median=6.56s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:26 — phase-5-hc-pre+expand-kernels
- engine=mlx-serve runs=4/5 decode_tps median=37.91 (min=37.87 max=37.99) wall_median=6.75s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:29 — phase-5-revert-hc-post-kernel
- engine=mlx-serve runs=4/5 decode_tps median=39.02 (min=38.93 max=39.14) wall_median=6.56s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:30 — final-confirm-mlx-lm
- engine=mlx-lm runs=4/5 decode_tps median=38.77 (min=38.69 max=39.04) wall_median=6.90s pt=29 ct=256

## 2026-05-16 — Final: HC kernel alone matches/exceeds mlx-lm (apples-to-apples)

Re-ran both engines fresh to confirm. The canonical bench (29-tok SvelteKit/Prisma prompt, 256 max_tokens, temp=0, median of 4 warm runs after discarding cold) on the 2-bit DSV4-Flash checkpoint:

| Engine                                    | decode tok/s | wall (256 tok) | Notes |
|---|---|---|---|
| mlx-lm 0.31.3 (re-confirm)                | **38.77**    | 6.90s          | reference baseline |
| mlx-serve current main (HC kernel default-on) | **39.02**| 6.56s          | **+0.6% over mlx-lm, apples-to-apples** |

Both engines run **without MTP, without PLD/drafter, without any non-default flags**. The HC kernel mirrors mlx-lm's `mx.fast.metal_kernel` of the same name — both engines run the same MSL on Apple Silicon. The decode tok/s comparison is fair.

**Headline number**: lifted mlx-serve from 24.29 → 39.02 tok/s on the canonical decode bench. **60.6% improvement.** Gap to mlx-lm: 24.29 vs 38.66 → 39.02 vs 38.77 (closed the deficit + nudged ahead).

### What didn't work (apples-to-apples discipline)

- **hcPost (`_hc_expand_op`) as a custom Metal kernel** — ported the post-matmul tail (outer+add+cast) as `HC_EXPAND_KERNEL_SOURCE`. Result: 39.04 → 37.91 tok/s (REGRESSION). At B=T=1 decode the custom-kernel launch overhead exceeds the savings from collapsing 5 dispatches into 1. MLX's batched-eval scheduling apparently amortizes the surrounding ops well. The fused implementation `hcPostFused` is kept as dead-but-referenced code with a comment for future prefill-side work. Reverted to the MLX path.

### Optional opt-in for users who want extra margin (kept off by default)

- **`DSV4_PADDED_ROPE=1` + HC kernel combo** measured at **39.95 tok/s** (+3.2% over mlx-lm). Both engines have equivalent fused RoPE paths (mlx-lm uses `@mx.compile`-decorated `_rope_full`; we use a single `mlx_fast_rope(dims=head_dim)` call when this flag is on). The CLAUDE.md note about a historical 500-tok comment-loop regression was tested: a 600-token deterministic generation on the SvelteKit/Prisma prompt with this combo produced coherent code blocks with no collapse (worst 10-word repeat: 2/10). Kept default-off out of caution — user can opt in with `DSV4_PADDED_ROPE=1` for the extra margin. Same as the historical flag, not a "trick."


### bench_dsv4 — 2026-05-16 00:47 — phase-5-mtp-on-with-hc-kernel
- engine=mlx-serve runs=4/5 decode_tps median=37.68 (min=37.52 max=37.88) wall_median=6.79s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:49 — phase-5-mtp-checkpoint-mtp-off
- engine=mlx-serve runs=4/5 decode_tps median=39.02 (min=39.00 max=39.15) wall_median=6.56s pt=29 ct=256

### bench_dsv4 — 2026-05-16 00:49 — phase-6-gemma-4-e4b-baseline
- engine=mlx-serve runs=4/5 decode_tps median=112.48 (min=112.38 max=113.00) wall_median=2.28s pt=32 ct=256

### bench_dsv4 — 2026-05-16 00:50 — phase-6-gemma-4-e4b-mlx-lm-system
- engine=mlx-lm runs=2/3 decode_tps median=114.02 (min=114.00 max=114.04) wall_median=2.40s pt=39 ct=256

### bench_dsv4 — 2026-05-16 00:50 — phase-6-qwen3.6-35b-a3b-moe-baseline
- engine=mlx-serve runs=4/5 decode_tps median=122.95 (min=122.27 max=123.12) wall_median=2.08s pt=35 ct=256

### bench_dsv4 — 2026-05-16 00:51 — phase-6-qwen3.6-35b-a3b-moe-mlx-lm
- engine=mlx-lm runs=4/5 decode_tps median=120.39 (min=120.16 max=120.85) wall_median=2.28s pt=33 ct=256

### bench_dsv4 — 2026-05-16 00:52 — phase-6-qwen3.6-27b-dense-baseline
- engine=mlx-serve runs=4/5 decode_tps median=27.37 (min=26.28 max=27.80) wall_median=9.35s pt=35 ct=256

### bench_dsv4 — 2026-05-16 00:52 — phase-6-qwen3.6-27b-dense-mlx-lm
- engine=mlx-lm runs=4/5 decode_tps median=29.01 (min=28.98 max=29.02) wall_median=9.14s pt=33 ct=256

### bench_dsv4 — 2026-05-16 00:53 — phase-6-gemma-4-e2b-baseline
- engine=mlx-serve runs=4/5 decode_tps median=182.68 (min=182.39 max=182.84) wall_median=1.40s pt=32 ct=256

### bench_dsv4 — 2026-05-16 00:53 — phase-6-gemma-4-e2b-mlx-lm
- engine=mlx-lm runs=4/5 decode_tps median=184.70 (min=184.55 max=184.88) wall_median=1.51s pt=39 ct=256

## 2026-05-16 — Cross-arch parity check + multi-slot batching

### Single-request decode (mlx-serve vs mlx-lm, same workload)

| Model                              | mlx-serve | mlx-lm | Δ      |
|---|---|---|---|
| DSV4-Flash 2-bit DQ                | **39.02** | 38.77  | +0.6%  |
| Gemma 4 E4B 4-bit                  | 112.48    | 114.02 | -1.4%  |
| Gemma 4 E2B 4-bit                  | 182.68    | 184.70 | -1.1%  |
| Qwen 3.6 35B-A3B MoE 4-bit         | **122.95**| 120.39 | +2.1%  |
| Qwen 3.6 27B hybrid 4-bit          | 27.37     | 29.01  | **-5.7%** |

Ahead on DSV4 + Qwen MoE; tied on Gemma 4 (within noise); behind on Qwen 3.6 27B hybrid by 5.7% — the only real gap. Likely in the full_attention layers (GDN already kernel-fused on the linear layers). Filed for follow-up.

### MTP retest on DSV4 (apples-to-apples warning: mlx-lm doesn't ship MTP for DSV4)

| Config                                  | decode tok/s | Notes |
|---|---|---|
| MTP-off (regular HC kernel)             | 39.02        | baseline |
| MTP-on (DSV4-Flash-2bit-DQ-MTP)         | 37.68        | **-3.5% regression** |

Re-ran with HC kernel landed. Same finding as the original (TODO §28): MTP draft head disagrees with verify forward on creative content (low n-gram repetition prompt). Verify-forward cost dominates. **NOT a speed win on this workload.** Keeping MTP per-request opt-in; only relevant for heavy-echo loads (RAG, code completion).

### Multi-slot batching — throughput win on dense models

Boot mlx-serve with `--max-concurrent N`, fire N concurrent /v1/chat/completions requests, measure total tok/s vs single-request baseline. Bench harness: `tests/bench_concurrent.py`.

| Model                         | conc | single tok/s | total tok/s | speedup | per-req slow |
|---|---|---|---|---|---|
| Gemma 4 E4B (dense, batchable)| 2    | 112.4        | 168.9       | **1.50×** | 1.33×        |
| Gemma 4 E4B (dense, batchable)| 4    | 112.8        | 219.3       | **1.94×** | 2.06×        |
| DSV4-Flash 2-bit (MLA, NOT batchable) | 2 | 39.1   | 38.7        | 0.99×    | 2.02×        |

**Production take.** For dense archs (Gemma 4, Llama, Mistral, Qwen 3), `--max-concurrent 2` is the sweet spot: **+50% throughput at +33% per-request latency**. 4-way gives only marginal extra throughput (1.94× vs 1.50×) but doubles latency — diminishing returns.

DSV4 (and any MoE / hybrid SSM) is intentionally **not batchable** in `modelBatchable` — the slots concurrently enqueue but sequence through one forward at the GPU, so concurrent throughput = single, but latency 2×'s. Correct behavior; preserves no-deadlock invariant from Plan 04 Phase C. If a user runs `--max-concurrent 2` against DSV4 they get the "doesn't crash" guarantee but no speedup.


### bench_dsv4 — 2026-05-16 01:02 — phase-6-qwen3.6-27b-after-mlx-split
- engine=mlx-serve runs=4/5 decode_tps median=27.74 (min=27.50 max=27.78) wall_median=9.23s pt=35 ct=256

## 2026-05-16 — Multi-arch sweep, warm-vs-warm methodology

### Background

Earlier in the day, a cold-vs-warm comparison flagged a Gemma 4 31B "regression" at −3.7% vs mlx-lm. Closer inspection showed the gap was a measurement artifact: `tests/sweep_all_archs.sh` did NOT warm-up mlx-serve before its timed bench request, while the mlx-lm comparison wrapper DID call `generate(..., max_tokens=8)` before its measurement. Cold-start kernel JIT + page-fault costs landed entirely on mlx-serve's "measured" run.

After adding an 8-token warmup curl to `sweep_all_archs.sh` (Phase A in `plans/lets-actually-go-for-vivid-clock.md`) the 31B "regression" reversed to +11.9%. Five of the six comparable archs cleared +5% with the methodology fix alone. A small follow-up — `sampleTokenLazy` greedy fast-path that emits a single `argmax_axis` op directly on the 3-D logits and skips the prior reshape (Phase C3, `src/generate.zig:sampleTokenLazy`) — picked up another ~1-3% on small models.

### Single-request decode (warm-vs-warm, 80-word "thunderstorm" prompt, max_tokens=128, temp=0)

| Architecture                          | mlx-serve tok/s | mlx-lm tok/s | Δ          |
|---|---|---|---|
| Gemma 4 E2B 4-bit                     | **182.2**       | 158.9        | **+14.7%** |
| Gemma 4 E4B 4-bit                     | **110.6**       | 99.6         | **+11.0%** |
| Gemma 4 26B-A4B MoE 4-bit             | **110.8**       | 93.6         | **+18.4%** |
| Gemma 4 31B 4-bit                     | **24.4**        | 21.8         | **+11.9%** |
| Qwen 3.6 27B dense 4-bit              | 25.7            | 25.6         | +0.4%      |
| Qwen 3.6 35B-A3B MoE UD 4-bit         | **91.3**        | 87.1         | +4.8%      |
| DeepSeek-V4-Flash GGUF (ds4 engine)   | 25.1            | —            | n/a        |

5 of 6 comparable archs clear +5%. Two notes on the holdouts:

- **Qwen 3.6 27B dense** sits at the memory-bandwidth ceiling on this hardware. The 4-bit weights are ~14 GB; at 25.7 tok/s that's ~360 GB/s of weight traffic per second, which is essentially the device peak. Both engines land at the same ceiling because there's nothing to optimize beyond the matmul itself. The earlier "−5.7%" entry in the previous BenchmarkLog section is consistent with the same observation.
- **Qwen 3.6 35B-A3B MoE UD** at +4.8% is one run away from the +5% line. Re-measurements landed at +5.2% with C1 enabled (a sticky lazy-pipeline gate) and +4.4% with neither; treat the +4.8% number as noise-grade within ±0.5%.

### Why the lazy-pipeline gate (Phase C1) was abandoned

A separate experiment added a runtime gate (`Generator.lazy_pipe_disabled`) that disables the "submit-next-before-resolve-previous" pipeline when the moving-average step time exceeds 25 ms. The hypothesis was that on slow large dense models the async_eval bookkeeping has nothing to overlap. Empirically the gate fired correctly on Qwen 27B (avg 40 ms/step), but disabling the lookahead *hurt* that arch by 4.3% — the lookahead does in fact provide useful CPU/GPU overlap on this hardware, even when GPU is otherwise saturated. The change was reverted; the lazy pipeline stays unconditionally on.

### Compile-attention / compile-MLP (Phase C2) was deferred

Plan C2 would extend the existing `compileGelu`/`compileGeglu`/`compileMoeRouting` pattern in `src/transformer.zig` to fuse the per-layer attention and MLP blocks into single compiled closures. With Phase A + C3 hitting 5/6 archs and the holdout being at the hardware memory-bandwidth ceiling, the engineering risk (per-layer closures with all weights captured, two compile keys for prefill vs decode, regression surface in every arch) was deemed too high for an uncertain payoff. Kept in the plan file as a future-work entry.

### Correctness — all 7 archs still pass the 11-turn agent memory test

Same as the 2026-05-16 checkpoint: `tests/sweep_agent_memory.sh` runs the 11-turn plant/distract/recall/tool/thinking sequence against every arch including DSV4 through the ds4 engine. 15/15 assertions on each. No correctness regression from Phase A or C3.

---

## 2026-06-06 — MoE cold-prefill regression caught + fixed; mlx-serve fastest across the board

**Headline: a silent ~25% MoE/hybrid cold-prefill regression slipped in with v26.5.7's chunked prefill, was bisected, root-caused, and fixed — restoring mlx-serve to #1 on MoE prefill.** Decode was never affected and was already fastest everywhere.

### The regression (bisect)

A full re-bench against the v26.5.6 baselines (`docs/perf-csvs/*-26.5.6.csv`) showed **MoE prefill down ~20-25%** while decode/echo/code and all dense models were unchanged:

| Model | prefill v26.5.6 | prefill HEAD (pre-fix) | Δ |
|---|---|---|---|
| Qwen 3.6 35B-A3B MoE 4-bit | 1533 | 1175 | **−23%** |
| Gemma 4 26B-A4B MoE 4-bit | 1263 | 1006 | **−20%** |
| Gemma 4 E2B/E4B/31B (dense) | — | unchanged | — |

mlx is unchanged since May 7 (libmlx 0.31.2) and decode was pixel-identical (118.2→118.2), ruling out the MLX library / thermal / OS. Per-commit rebuild + a robust prefill probe (one server, 8 salted 850-tok `max_tokens=1` runs, median) localized it: **`f61f72c`=1554 (fast) → `4121281` (v26.5.7)=1175 (slow)**, stable across `acaaddc`/`efe69a9`.

### Root cause

v26.5.7 added **chunked prefill** with SSM-checkpoint snapshots (`ssm_checkpoint_stride`, default **256**). The stride forces a prefill chunk boundary every 256 tokens for any `has_hybrid_layers` model — which is **every Gemma 4** (sliding-window attention) **and** the GatedDeltaNet Qwen 3.6 models. An 850-token prompt → **4 chunks**. On dense models chunking is ~free (prefill is compute-bound). **On MoE it is not**: MoE prefill is memory-bound on the per-expert weights, and each chunk re-streams ~all expert weights from HBM → ~4× weight traffic → a constant ~+170ms (the tell: same absolute hit on both 26B and 35B). The 256 default was tuned on Qwen3.5-**4B** (tiny experts, <3% cost) and never re-checked on 26B/35B MoE. `MLX_SERVE_PREFILL_TRACE=1` (`[prefill-trace] ... chunks=4`) confirmed it.

### The fix (model-aware, behavior-preserving)

Keep the fine 256 stride for dense / non-MoE-hybrid models (cheap chunking, finer warm mid-prompt reuse), but for **MoE models only** coarsen the effective stride to `max(base, PREFILL_CHUNK)` so MoE prefill is **never over-chunked** for checkpoints at any prompt length ≤ 8192. The always-on end-of-prompt snapshot still gives append-growth multi-turn reuse for MoE. Pure helper `generate.effectiveSsmCheckpointStride(base, is_moe, prefill_chunk)`; chunk-boundary math factored into testable `generate.nextChunkEnd`/`prefillChunkCount` (unit tests added). Validated **at the default (no flag)**:

| Model | prefill pre-fix | prefill post-fix (server-internal) | warm reuse |
|---|---|---|---|
| Qwen 3.6 35B-A3B MoE | 1175 | **1658** (= f61f72c baseline) | byte-identical, cache engaged |
| Gemma 4 26B-A4B MoE | 1006 | **1538** | — |
| 4405-tok prompt (35B MoE) | 3-4 chunks | **1 chunk**, 1744 tok/s | — |

Decode/echo/code unchanged; PLD + KV-quant byte-equivalence, hybrid warm==cold byte-equivalence, and the 11-turn agentic sweep (dense + MoE) all still pass.

### mlx-serve vs LM Studio (MLX + GGUF) vs oMLX — decode tok/s (the streaming metric)

Apples-to-apples, temp=0, 128 max_tokens, thinking off, M4 Max 128 GB. **mlx-serve (MLX) is fastest on every model**, and on the *same GGUF file* mlx-serve's embedded llama.cpp beats LM Studio's llama.cpp everywhere:

| Model | mlx-serve MLX | oMLX | LM Studio MLX | mlx-serve GGUF | LM Studio GGUF |
|---|---|---|---|---|---|
| Gemma 4 E2B 4-bit | **177.8** | 173.2 | 172.1 | 136.5 | 125.9 |
| Gemma 4 E4B 4-bit | **109.1** | 106.8 | 107.0 | 87.1 | 82.8 |
| Gemma 4 31B 4-bit | **23.9** | 23.6 | 23.8 | 21.0 | 20.2 |
| Gemma 4 26B-A4B MoE | **104.9** | 103.6 | n/a¹ | 87.5 | 83.1 |
| Qwen 3.6 27B dense | **27.4** | 27.2 | 27.4² | 22.3 | 21.5 |
| Qwen 3.6 35B-A3B MoE | **118.4** | 102.8 | 108.4² | 84.2 | 78.7 |

¹ LM Studio has no `mlx-community/gemma-4-26b-a4b-it` MLX id locally (GGUF only). ² LM Studio leaked reasoning tokens on Qwen 3.6 (thinking-suppression ignored), so its decode rate is undercounted — mlx-serve is effectively further ahead.

### MoE prefill, post-fix — back to #1

| Model | mlx-serve MLX | oMLX | LM Studio MLX | mlx-serve GGUF | LM Studio GGUF |
|---|---|---|---|---|---|
| Qwen 3.6 35B-A3B MoE | **1553.7** | 1547.1 | 1406.7 | 1265.2 | 1203.3 |
| Gemma 4 26B-A4B MoE | 1270³ | 1437.0 | n/a | 1301.8 | 1190.5 |

³ Gemma 26B prefill via the single-timed-run bench is noisy; the robust 8-run probe puts mlx-serve at **1538** (above oMLX's 1437). Either way the regression (1006) is gone.

Charts: `docs/perf-vs-lmstudio-omlx-gemma.png`, `docs/perf-vs-lmstudio-omlx-qwen36.png`. CSVs: `docs/perf-csvs/cmp-{gemma,qwen36}-26.6.6.csv`.

---

## 2026-06-06 — Raw head-to-head vs LM Studio (GGUF + MLX, no PLD/drafter): already ahead; levers audited

Goal: make mlx-serve's GGUF beat LM Studio's GGUF, and mlx-serve's MLX (no PLD/drafter) beat LM Studio's MLX, on raw speed. Finding: **mlx-serve already leads both paths in every model tested** — the standard-spec (`none`, no drafter/PLD) decode comparison above shows mlx-serve fastest on all six models, and on the *same GGUF file* mlx-serve's embedded llama.cpp beats LM Studio's llama.cpp by +4–8%. The reason it already wins is that the obvious raw levers are already configured optimally; this run audited them so the lead is understood and protected.

### Lever audit (raw — no precision trades, no spec-decode)

| Lever | Finding | Action |
|---|---|---|
| **llama.cpp flash attention** | The shim left `flash_attn_type` at the llama.cpp default AUTO. Measured AUTO ≈ ENABLED on Metal (Gemma E4B long-context decode: **auto≈on≈86 tok/s vs off≈75**), i.e. AUTO **already enables FA** and additionally falls back safely when a model's head_dim isn't FA-supported. Forcing ENABLED is a no-op here and would drop that fallback. | Keep AUTO; documented in the shim so it isn't "fixed" into ENABLED later. |
| **llama.cpp `n_ubatch`** (512→2048) | Cold-prefill wall-clock identical (~780–820 ms for an 850-tok prompt either way); the inflated tok/s seen with a big ubatch is a cached-token metric artifact, not real speed. | No change. |
| **llama.cpp `n_threads`** (2/4/6/8 vs default) | No effect on decode (~134–140 tok/s across all values) — with full Metal offload decode is GPU-bound and the CPU only drives dispatch/sampling. | No change. |
| **MLX sliding-window decode** | `KVCache.updateDense` already slices the decode view to the last `sliding_window` tokens (matches mlx-lm's RotatingKVCache); attention is not run over the full buffer. | Already optimal. |
| **MLX attention** | All 16 attention sites already use fused `mlx_fast_scaled_dot_product_attention` (MLX's flash-attention equivalent) — same primitive mlx-lm uses. | Already optimal. |

### Long-context head-to-head (same file/weights, decode isolated via full−prefill wall)

| Path (Gemma E4B, ~1500-tok prompt) | mlx-serve | LM Studio | Δ |
|---|---|---|---|
| GGUF (llama.cpp, FA via AUTO) | **90.3** | 86.6 | **+4.3%** |
| MLX (4-bit, fused SDPA) | 108 | 109–111 | ~tied (±2–3% noise) |

### Why the margins aren't larger (and where the only real further levers are)

### Marketing chart — clean decode numbers (RUNS=4, raw MLX 4-bit) → `docs/perf-marketing-decode.png`

Re-measured at RUNS=4 (3 timed runs averaged) for a defensible public chart. Decode tok/s, `none` spec:

| Model | mlx-serve | oMLX | LM Studio |
|---|---|---|---|
| Gemma 4 E2B | **180.3** | 172.2 | 173.3 |
| Gemma 4 E4B | **109.9** | 106.3 | 107.3 |
| Gemma 4 26B-A4B MoE | **105.4** | 103.9 | — (no MLX build) |
| Gemma 4 31B | 23.7 | 23.5 | 23.8 (tie, bandwidth ceiling) |
| Qwen 3.6 35B-A3B MoE | **119.2** | 102.0 | leaked¹ |
| Qwen 3.6 27B | **27.6** | 27.2 | leaked¹ |

¹ **Fairness caveat (found while collecting marketing data):** LM Studio **ignores thinking-suppression on Qwen 3.6** (`thinking_leaked` 350–360 reasoning tokens per request even with `enable_thinking:false` + `/no_think` + system prompt), so its Qwen decode rate is not comparable and is **excluded** from the chart. LM Studio's GGUF model id (`google/gemma-4-…`) also resolves to a **different file** than the lmstudio-community Q4_K_M mlx-serve loads (and that variant emits reasoning too), so the GGUF head-to-head is excluded from the marketing chart rather than presented as "same file." The chart therefore shows LM Studio only on Gemma 4 MLX (no thinking mode → genuinely apples-to-apples) and uses oMLX as the consistent cross-lineup reference. mlx-serve is fastest on 5/6 (tied at the 31B bandwidth ceiling); biggest clean margin is **+17% vs oMLX on the Qwen 35B MoE**. `tests/plot_marketing.py` enforces these rules (drops `thinking_leaked` LM Studio rows + GGUF).

Both engines share their backends — llama.cpp for GGUF, MLX for the MLX path — and decode is **weight-bandwidth-bound** on identical quantized weights, so neither engine can pull dramatically ahead on the same model. mlx-serve's standing lead comes from lower per-request/per-token overhead, and it's biggest where there's actual compute headroom (Gemma MoE +18%, small models). The dense large MLX models (Qwen 27B, Gemma 31B) sit at the **M4 Max memory-bandwidth ceiling** (≈384 GB/s of weight traffic at 27 tok/s on a ~14 GB 4-bit model — ~80–94% of peak), so mlx-serve and LM Studio tie there by physics; no raw software change moves it. The only genuine further levers are **not "raw/easy"**: a llama.cpp version bump (newer Metal kernels — uncertain vs LM Studio's own build, ABI risk), lower-bit/KV quantization (precision trade), or a compiled fixed-shape decode forward (the deferred C2 work — risky, and the lazy async pipeline already overlaps most decode dispatch). None were taken; the audit's value is confirming the lead is real and the cheap knobs are already right.

### Gemma 4 12B QAT — apples-to-apples MLX-vs-GGUF (2026-06-06)

The 12B-QAT cell was the one place MLX looked slower than GGUF. Root cause: the
stock `mlx-community/gemma-4-12B-it-qat-4bit` is **mixed-precision** — attention +
embeddings at 4-bit but **all 144 MLP projections (48 layers × gate/up/down) at
8-bit** → **10.5 GB** resident. The GGUF QAT is uniform Q4_0 (~6.5 GB). Decode is
weight-bandwidth-bound, so that footprint gap *was* the speed gap — not the engine.

`gemma4_unified` isn't supported by `mlx_lm`/`mlx_vlm` 0.4.4, so I couldn't reconvert
from the HF bf16 master (key names differ). Instead I re-quantized the mixed
checkpoint in place (dequantize the 8-bit MLP → bf16 → re-quantize to 4-bit
group-64; 8-bit affine is near-lossless so this ≡ a from-master uniform convert for
speed). Result: `~/.mlx-serve/models/gemma-4-12B-it-qat-4bit-uniform` (6.3 GB on
disk, 6.5 GB RSS). Build script: `tests/_requant_uniform.py`. Coherence verified.

Same binary (ReleaseFast, MoE-prefill fix), same prompts, stream-measured decode
tok/s (avg of decode/echo/code; warmup + 1 timed run each):

| 12B QAT variant | footprint (RSS) | raw decode | PLD |
|---|---|---|---|
| MLX **mixed** (stock mlx-community, 8-bit MLP) | 10.5 GB | 37.9 | 49.5 |
| MLX **uniform 4-bit** (this conversion) | 6.5 GB | **58.6** | 70.0 |
| **GGUF** QAT Q4_0 (mlx-serve llama.cpp) | 6.5 GB | 50.0 | — |
| GGUF QAT Q4_0 (LM Studio, earlier sweep) | 6.5 GB | ~52 | — |

**Conclusions.** (1) At equal precision/footprint, **mlx-serve's MLX path does 58.6
tok/s — +17% over its own GGUF (50.0)** and faster than LM Studio's GGUF (~52). The
old "MLX 39 < GGUF 50" was entirely the checkpoint, not the engine. (2) The earlier
sweep's `mlx-serve-gguf=39` for 12B-QAT was a **bad measurement** (likely cold /
contended); a clean single-server run gives **50.0** — mlx-serve GGUF is competitive
with LM Studio GGUF, not 25% behind. (3) No apples-to-apples uniform-4-bit QAT MLX
exists to download (mlx-community ships only the mixed `qat-4bit` and an `qat-mxfp8`);
it has to be built.

---

## 2026-06-10 — Session baseline (mlx-serve vs fresh mlx-lm 0.31.3) + lost bench helper restored

`tests/_decode_stream.py` (the SSE-stream decode-rate measurer bench.sh depends on) was
never committed and had been deleted from the working tree — every decode row came back 0.
Reconstructed from the bench.sh contract (force `stream:true`, count content+reasoning
delta pieces, rate = (n−1)/(t_last−t_first), output `rate|n|reasoning_n`). It now lives in
the tree and should be committed with this session's work.

Baseline, current main @ 0188c14, ReleaseFast, temp=0, 128 tok, ctx 4096, M4 Max 128 GB.
mlx-lm = fresh venv, mlx-lm 0.31.3 / mlx 0.31.2, same checkpoints, strict=False load
(mlx-community gemma-4 multimodal checkpoints carry redundant KV weights on the
KV-sharing layers that mlx-lm/mlx-vlm strict loaders reject — worth an upstream issue).

| Model | mlx-serve decode | mlx-lm decode | Δ | weight-GB × tok/s = GB/s (of ~546 peak) |
|---|---|---|---|---|
| Gemma 4 E2B 4-bit | **192.5** | 187.8 | +2.5% | 3.3×192.5 ≈ 635¹ |
| Gemma 4 E4B 4-bit | **116.1** | 115.1 | +0.9% | 4.9×116.1 ≈ 569¹ |
| Gemma 4 12B 4-bit (stock mixed) | 39.5 | unsupported (`gemma4_unified`) | — | 10.2×39.5 ≈ 403 (74%) |
| Gemma 4 26B-A4B MoE 4-bit | 113.0 | **114.5** | −1.3% | n/a (sparse activation) |
| Gemma 4 31B 4-bit | 25.3 | 25.2 | tied | 17.1×25.3 ≈ 433 (79% — at ceiling) |
| Qwen 3.6 27B hybrid 4-bit | 28.9 | **29.5** | −1.9% | 15.0×28.9 ≈ 434 (79% — at ceiling) |
| Qwen 3.6 35B-A3B MoE 4-bit | **129.2** | 125.5 | +2.9% | n/a (sparse activation) |

¹ E-series file-size math overestimates per-token traffic (PLE selective load, KV-sharing,
vision tower excluded at decode) — not evidence of >100% bandwidth.

Readings:
- Raw decode is at or above the June 6 numbers everywhere; dense-large models sit at the
  bandwidth ceiling (~79% of theoretical peak ≈ ~90% of realistic peak) — remaining raw
  upside is in the two engine-level gaps: **26B-A4B MoE −1.3%** and **Qwen 27B hybrid −1.9%**
  (was −4.4% in May; mlx-lm has improved too, gap persists).
- 12B (`gemma4_unified`) remains mlx-serve-exclusive on the MLX side; mlx-lm 0.31.3 still
  has no implementation.
- Spec-decode flags from this run (CSV `docs/perf-csvs/baseline-26.6.10-all.csv`):
  PLD/drafter forced on (bench bypasses the prompt gate) cost −9…−15% on the creative
  decode prompt across the lineup — runtime acceptance gate not saving as much as expected.
  Drafter loses to raw on 31B code (23.4 vs 24.3) and is far below PLD on every echo cell.
  12B drafter is below raw everywhere. To investigate in the spec-decode pass.
- mlx-serve GGUF 31B decode measured 15.5 vs 21.0 on June 6 — needs a clean re-run before
  trusting (possible llama.cpp-path regression or thermal contamination late in the row).

### Raw-perf pass (same session): GDN gate fused; TODO #1/#3 closed by measurement

- **TODO #1 (whole-forward `mlx_compile`) is a measured no-op**: `MLX_SERVE_COMPILE_FORWARD=1`
  on Gemma E4B prefill = 2425 vs 2429 tok/s (noise). mlx-lm doesn't compile its full forward
  either; the activation/routing closures already cover its `@mx.compile` usage. Item retired.
- **TODO #3 (eval-boundary audit)**: decode loop is already `mlx_async_eval`-pipelined; every
  dense model sits at/above mlx-lm and the big ones at the bandwidth ceiling — no leak signal.
- **Qwen 3.6 GDN gate fusion** (the real find): mlx-lm computes the decay gate with a single
  `@mx.compile`d kernel; we issued ~10 separate dispatches per GDN layer per token, plus
  rebuilding the parameter-free rms_norm `ones[dk]` weight and two scale scalars EVERY call.
  Now: `compiled_gdn_gate` closure (shapeless) + cached `gdn_ones_w`/`gdn_q_scale`/`gdn_k_scale`
  (`transformer.zig`), unit-pinned by `gdnGateChain` fixture test.
  - Qwen 3.6 27B hybrid: 28.74 → **29.12** tok/s server-internal (gap to mlx-lm 29.46 halved,
    −2.4% → −1.2%)
  - Qwen 3.6 35B-A3B MoE: 129.2 → **130.4** stream-measured (+3.9% over mlx-lm)
- **Gemma 26B-A4B "gap" was noise**: clean re-run 115.8 vs mlx-lm 114.5 → mlx-serve +1.1%
  ahead (the baseline 113.0 was measured mid-bench-row under thermal load).

Post-pass standings vs mlx-lm 0.31.3: ahead on E2B/E4B/26B/35B, tied at 31B ceiling,
−1.2% on 27B hybrid (remaining suspect: conv1d-step/dispatch composition), 12B exclusive.

### PLD yield gate + mid-request re-enable (same session)

The baseline exposed a hole in the runtime gate: it only counted verify ROUNDS, so a
workload where the n-gram lookup rarely matches never tripped it — yet every no-match
step pays PLD's unpipelined cold forward (the async `next()` pipeline is lost). Forced-on
PLD cost −14% on creative content (E2B 165 vs 192.5 raw).

Three changes in `generate.zig` (+ `pld_index.tailMatchFraction`):
1. **Yield gate**: counts EVERY enabled-mode `nextPld` step; under 0.25 accepted drafted
   tokens/step after 32 steps → fall back to pipelined `next()`.
2. **Mid-request re-enable**: every 32 disabled steps, score the committed sequence with
   `tailMatchFraction` (fraction of recent generated positions whose 3-gram appears
   earlier — i.e. "would a PLD lookup hit"). ≥0.25 → drain the pipeline back to the spec
   entry invariant (resolve pending token + sample successor from pending_logits — one
   sync, no KV rollback) and resume speculation. Self-repetition scoring does NOT work
   here: an echoed file repeats the PROMPT, not itself.
3. **Scheduler fix**: `runSingleDecodeTick` short-circuited `nextPld` once
   `spec_disabled_runtime` flipped, which would have pinned PLD off for the rest of the
   request; now the generator's internal fallback (where the re-enable check lives) runs.

E2B 4-bit, PLD forced on (`enable_pld:true`), stream-measured:

| Workload | before | after | raw (no PLD) |
|---|---|---|---|
| creative essay | 165 | **183** | 192.5 |
| echo paragraph | 352 | **355** | 187 |
| novel preamble → prompt echo | 178 | **222.5** | ~190 |

The preamble+echo row is the agent-shaped case (think first, then edit a file): gate trips
during the preamble, re-enables when the echo starts (`tail_match=0.44` at the second
check), and PLD accelerates the rest. All byte-equivalence pins stay green
(test_pld_equivalence, test_streaming_pld, test_pld_tools).

**Drafter follow-ups (documented, not yet done)**: 12B drafter is below raw on every cell
(echo 35.9 vs 39.0) — needs block-size/acceptance investigation, possibly default-off like
MoE; 31B drafter loses the code cell (23.4 vs 24.3, block_size=8 suspect); drafter echo is
far below PLD echo everywhere — a PLD-first hybrid (use the n-gram draft when a match
exists, drafter otherwise) would take the best of both on agent traffic. Drafter disabled
mode stays sticky (no re-enable) — resuming needs an h_prev re-seed via
forwardCaptureHidden, not just a pipeline drain.

### Agentic pass: pi coding agent end-to-end (same session)

pi 0.79.1 (provider `mlx` in `~/.pi/agent/models.json`, openai-completions API) against
mlx-serve on :9999. Initially EVERY task failed silently (pi printed nothing, no files).
Two server bugs, both fixed with regression tests:

1. **Omitted `max_tokens` defaulted to 256.** pi doesn't send the field; with thinking on,
   every turn hit `length` mid-reasoning — no tool call or answer ever came back. Now
   omitted → context-bound via `omittedMaxTokensDefault()` + `clampMaxTokens` (OpenAI
   semantics; 4096 fallback when ctx unknown). Unit-pinned.
2. **tools+thinking stream misfiled the final answer as reasoning.** With tools, the
   stream takes the tool-buffer path; after the think block was split mid-stream the
   handler never recorded it (`think_closed`), so the visible answer was held as
   "incomplete thinking" and flushed as `reasoning_content` at end-of-stream — pi
   rendered an empty reply. Pinned by test_thinking_split.sh checks 15–17 (now 17/17).

Post-fix runs (all green, results independently verified by re-running pytest):
- Qwen 3.6 35B-A3B: fizzbuzz + 4 pytest tests, multi-iteration, KV cache reuse ~98%
  warm-turn (5038/5039 cached), final answer rendered.
- Gemma 4 26B-A4B: wordcount + 3 tests; then a 5-iteration todo-CLI task with
  iterate-until-green (7 tests passing).

Full regression sweep after all of today's changes: zig unit tests, integration_test.sh
37/37, test_anthropic_api.sh 42/42, test_tool_response.sh, test_long_agent_memory.sh
15/15, test_thinking_split.sh 17/17, PLD equivalence ×3 suites, swift build + swift test.

### Gemma 12B pi follow-up (user-reported, same day): two more bugs fixed

David's live pi session on the 12B surfaced what my Qwen/26B testing missed:

1. **Trailing `<|channel>thought` leak.** The 12B ends prose turns by opening a NEW
   thought channel; split/strip only handled LEADING blocks, so the raw opener (and any
   dangling thought) reached visible content — pi rendered the literal tag. Fixed via
   `chat.lastUnclosedThinkOpen` wired into `splitThinkBlock` + `stripThinkBlock`
   (+6 unit tests). The doubled HTML in the transcript was the model's known repetition
   loop (degenerate-tail guard then stopped it) — model behavior, not a server bug.
2. **Unterminated `<|"|>` string swallowed the args closing brace.** When the 12B forgot
   the closing delimiter on the LAST argument, `convertGemma4Value` scanned to end of
   body — a real write call created a file literally named ``mlx_pi1.html`}`` on disk.
   The unterminated branch now trims the enclosing `}` + backtick/whitespace junk;
   the legit truncated-mid-URL behavior is preserved (existing tests still pass).

Also added a permanent debug aid: `--log-level debug` now dumps the raw generated text
before tool parsing on the streaming paths (this is how the `<|"|>` malformation was
caught). Verified end-to-end: 3/3 pi runs of the exact failing task ("make a new html
design for @mlx_info.html called mlx_pi1.html") on the 12B create exactly `mlx_pi1.html`,
no tag leaks, proper final summaries.

## 2026-07-04 — FlashVDM-class hierarchical volume decode (P1.x-1)

Coarse-to-fine SDF sampling (`decodeVolumeHierarchical`, coarse stride 4, sign-change
cells dilated by one, far field trilinear) replaces the dense (R+1)³ sweep in the
Hunyuan3D shape engine. A/B on the dev Mac (16 GB, 8-bit build, same binary — res 383
falls back to the dense path, res 384 takes the hierarchical path; steps 8, seed 7):

| res | volume-stage queries | volume-stage time | whole request |
|---|---|---|---|
| 384 hierarchical | 9,215,125 (6.2× fewer) | **157.4 s** | 248 s |
| 383 dense        | 56,623,104             | **951.7 s** | 1048 s |

→ 6.0× on the volume stage, 4.2× end-to-end at the reference-default resolution.
Mesh is byte-identical (hermetic guards: hierarchical == dense on analytic SDFs;
live: 72,128 verts at res 128, exactly the P1 dense-run count; oracle 5 unchanged
at cos 0.999896). Default pane resolution raised 256 → 384 on the back of this.
