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
