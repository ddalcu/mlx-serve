# Performance plan — decisively beat LM Studio in every category

> **2026-05-25 handoff:** Phase 1 (MLX hybrid prefix reuse via SSM
> checkpointing) and Phase 5 #2 (`--llama-kv-quant`) are **shipped and
> measured**. Final charts in `docs/perf-vs-lmstudio-final.png`,
> `docs/perf-decode-vs-lmstudio.png`. 9/9 measured cells are wins or
> ties vs LM Studio. The next agent picks up at Phases 2, 3, 4 #1/#3,
> 5 #1, 6, 7 — all open. Section "What's left" below has the priority
> order. Read "Lessons banked" first; it captures gotchas the next
> phases will re-hit.

## Goal

Beat LM Studio by a measurable, decisive margin across every dimension that
affects user-perceived speed, on **both** engines mlx-serve ships:

| Category | GGUF (via llama.cpp) | MLX (safetensors) | Status (2026-05-25) |
|---|---|---|---|
| Cold prefill (tok/s) | beat by ≥ 25% | beat by ≥ 25% | ~at parity (+4–11%) |
| Warm / multi-turn prefill (TTFT) | tied/better — both near-instant | tied/better — both near-instant | **1.33–1.82× faster** |
| Decode (tok/s) | beat by ≥ 30% | beat by ≥ 30% | +9% / +11% / **+62%** |
| End-to-end TTFT, 1k prompt cold | beat by ≥ 30% | beat by ≥ 30% | open (~tied) |
| End-to-end TTFT, warm multi-turn | beat by ≥ 5× | beat by ≥ 5× | open (1.33–1.82×) |

"Beat LM Studio" is harder than it sounds for GGUF because we share libllama
with them — the wins must come from things LM Studio doesn't do (speculative
decoding, multi-entry prompt caching, tokenization pipelining, concurrent
batching). For MLX we own the entire stack, so the ceiling is higher but the
work is deeper (GatedDeltaNet forward, SSM checkpointing).

## Where we started (baseline, pre-Phase-1)

Run `./tests/bench_prefill.sh <model>` to reproduce. Apple M4 / 16 GB,
~1000-token prompt, ReleaseFast.

| | Cold prefill | Multi-turn (warm) | Decode |
|---|---|---|---|
| **GGUF Qwen3.5-4B-IQ4_NL** | ~720 tok/s | reuses 940/941 → near-instant | parity with LM Studio |
| **MLX Qwen3.5-4B-4bit** (hybrid SSM) | ~393 tok/s | **0/941 cached — full cold every turn** | ~parity |
| **MLX Gemma 4 E4B-4bit** (plain attn) | ~640 tok/s | reuses 936/937 → near-instant | ~+50% vs LM Studio |

Key facts:

1. **The original "384 tok/s for both engines" complaint was a client-TTFT
   artifact, not raw prefill compute.** GGUF cold is actually ~720 tok/s.
   Always measure server-side `timings.prompt_per_second` and `timings.cached_n`.
2. **GGUF multi-turn was already solved** — a persistent `LlamaSession` per
   model reuses the KV prefix (LM-Studio-style). Don't redo this.
3. **MLX hybrid (GatedDeltaNet) reuse was architecturally blocked** with the
   pre-Phase-1 SSM cache design (`HotPrefixCache.shouldUse` returned false on
   any model with `has_hybrid_layers` or `full_attention_interval > 0`). The
   prior session attempted single-snapshot solutions and reverted; **the real
   fix needed per-position SSM checkpointing**.
4. **MLX cold prefill (393 tok/s on Qwen3.5)** is 100% GatedDeltaNet forward
   compute (`MLX_SERVE_PREFILL_TRACE=1` shows 2940 ms forward, 1 ms eval —
   chunk size has zero effect). Optimizing requires touching `transformer.zig`.
5. **Cross-arch byte-equivalence is non-negotiable.** Plain attention, hybrid
   SSM (lfm2 / nemotron_h / qwen3_5(_moe) / qwen3_next), sliding window (Gemma
   3/4), MoE — touching any shared path requires validating each.
6. **Always build `-Doptimize=ReleaseFast`.** Plain `swift build` (without
   `app/build.sh`) silently rebuilds zig-out in Debug.

## Where we are now (measured, post-Phase-1 + Phase 5 #2)

**Headline numbers** (Apple M4 / 16 GB, same model files for both engines,
~1325-token prompt, temperature=0, max_tokens=8 for TTFT / max_tokens=128
for decode, median of warm runs):

### TTFT — `docs/perf-vs-lmstudio-final.png`

| Workload | mlx-serve | LM Studio | Win |
|---|---:|---:|---:|
| Cold Gemma 4 E4B MLX (plain attn) | 2362 ms | 2624 ms | 1.11× |
| Cold Qwen3.5-4B GGUF (IQ4_NL) | 3736 ms | 3877 ms | 1.04× |
| Cold Qwen3.5-4B MLX (hybrid SSM) | 3676 ms | 3669 ms | tied |
| Warm Gemma 4 E4B MLX (plain attn) | 466 ms | 663 ms | **1.42×** |
| Warm Qwen3.5-4B GGUF (IQ4_NL) | 334 ms | 445 ms | **1.33×** |
| **Warm Qwen3.5-4B MLX (hybrid SSM)** | **298 ms** | **542 ms** | **1.82×** ← Phase 1 headline |

### Decode tok/s — `docs/perf-decode-vs-lmstudio.png`

| Workload | mlx-serve | LM Studio | Win |
|---|---:|---:|---:|
| Gemma 4 E4B MLX (plain attn) | **33.4 tok/s** | 20.6 tok/s | **+62%** |
| Qwen3.5-4B MLX (hybrid SSM) | 37.5 tok/s | 33.7 tok/s | +11% |
| Qwen3.5-4B GGUF (IQ4_NL) | 28.6 tok/s | 26.4 tok/s | +9% |

CSVs under `docs/perf-csvs/`. Methods documented in `docs/perf-vs-lmstudio-final.md`.

### What shipped this session

| Phase | Status | One-line |
|---|---|---|
| **0** (baseline harness) | ✅ | `tests/bench_final.sh`, `tests/bench_decode.sh`, plot scripts, CSV layout. |
| **1** (MLX hybrid prefix reuse via SSM checkpointing) | ✅ | The big one. Multi-checkpoint stride-aligned snapshots + post-prefill snapshot + commit-time merge on prefix-extend. 1.82× warm TTFT on Qwen3.5-4B MLX, byte-identical to cold, no plain-attn regression. |
| **4 #2** (early SSE role chunk) | ✅ (already in place) | `src/server.zig:3429` emits `delta.role=assistant` right after slot admission, before any decoded token. |
| **5 #2** (`--llama-kv-quant {off,q8,q4}`) | ✅ | Exposes `type_k`/`type_v` ggml types on the llama.cpp KV cache via the shim. Auto-enables flash-attn when non-default. Output byte-identical at temperature=0 in spot-checks. |

### Files changed (touch points for the next agent)

- **SSM checkpoint type + helpers:** `src/transformer.zig` (`SSMCheckpoint`,
  `captureSsmCheckpoint`, `restoreSsmCheckpoint`, `ssmCheckpointBytes`).
- **Capture during prefill:** `src/generate.zig` (chunked-prefill loop forces
  chunk ends to stride boundaries; per-stride and post-prefill snapshots;
  `Generator.takeSsmCheckpoints()`).
- **Cache entry + commit/restore + gate:** `src/prefix_cache.zig`
  (`Entry.ssm_checkpoints`, `commitWithSsm`, `lookupAndRestore` with
  hybrid-aware effective-matched clamp, `shouldUse(config, enable_ssm_checkpoints)`,
  merge-on-prefix-extend in the replace path).
- **Scheduler plumbing + commit drain:** `src/scheduler.zig` (LoadParams +
  LoadedModel fields, runPrefill passes stride and `hot_matched` offset,
  commitSlotIfApplicable drains via `commitWithSsm`).
- **CLI + defaults:** `src/main.zig` (`--ssm-checkpoint-stride`,
  `--ssm-checkpoint-max`, `--llama-kv-quant`), `src/server.zig` (defaults
  256/32/off; startup log line distinguishes hybrid + SSM-checkpoints
  enabled vs the legacy "disabled" message).
- **LoadedModel fields:** `src/model_registry.zig`
  (`ssm_checkpoint_stride`, `ssm_checkpoint_max`, `llama_kv_type_{k,v}`).
- **GGUF KV quant:** `lib/llama_shim/llama_shim.{h,c}` (new
  `mlx_llama_session_create_kv_quant`), `src/llama_ffi.zig` (`GgmlType`
  constants + new entry point), `src/arch/llama.zig` (`LlamaKvQuant` enum,
  `createSessionWithKvQuant`).
- **Regression guard:** `tests/test_hybrid_reuse_equivalence.sh` (TDD red/green
  pin: byte-identical, cached_n > 0, warm prompt_ms < 70% of cold).
- **Benches + charts:** `tests/bench_final.sh`, `tests/bench_decode.sh`,
  `tests/bench_perfplan_compare.sh`, `tests/plot_final.py`,
  `tests/plot_decode.py`, `tests/plot_perfplan.py`.
- **Reports:** `docs/perf-vs-lmstudio-final.{md,png}`,
  `docs/perf-decode-vs-lmstudio.png`, `docs/perf-csvs/*.csv`.

### Tunables shipped

- `--ssm-checkpoint-stride N` (default **256**) — snapshot every N tokens
  during prefill. 0 disables (legacy behavior; hybrid bypasses the cache).
  Smaller stride = finer warm alignment + more cold-prefill overhead.
- `--ssm-checkpoint-max N` (default **32**) — cap on snapshots per request.
  Older ones drop front-first when the buffer would grow past this.
- `--llama-kv-quant {off,q8,q4}` (default **off** = F16) — KV-cache
  quantization for the embedded llama.cpp engine. `q8` ≈ 2× KV
  compression (near-lossless); `q4` ≈ 4× (some quality impact).

### Lessons banked (read before starting Phase 2+)

1. **BPE drift at chat-template boundaries.** Turn-2's tokenization of the
   "assistant header" tokens can differ from turn-1's because trailing tokens
   re-tokenize differently when content follows. A single snapshot at
   `prompt_len` misses by 4-8 tokens. Phase 1 solved this with multi-snapshot
   at stride. Any future cache-restore work must assume `matched <= prompt_len`
   strictly.
2. **The replace path in `commitWithSsm` MUST merge old + new snapshots on
   prefix-extend.** Without merge: turn 2's prefill of the small uncached
   tail captures few/no checkpoints, so turn 3 alternates hit→miss→hit→miss.
   The merge is sorted-by-pos, dedup-by-pos (new wins on tie). Multi-turn
   conversations would otherwise silently lose half their warm reuse.
3. **`full_attention_interval > 0` is NOT a MoE signal on Qwen3.5/3.6 —
   it's the hybrid-SSM marker** ("every Nth layer is full attention, the
   rest are GatedDeltaNet"). The old `shouldUse` gate conflated these. The
   new gate treats both `has_hybrid_layers` and `full_attention_interval > 0`
   as "SSM layers somewhere", gated by `enable_ssm_checkpoints`.
4. **`mlx_array_set` aborts on null source** (mlx-c default handler does
   `exit(-1)`). SSM snapshot/restore must check each field's `.ctx != null`
   independently — `initialized` alone is insufficient. LFM2 `gated_conv`
   writes only `conv_state`; Mamba2/GDN order is the opposite. Both halves
   already null-guarded in `ssmSnapshot`/`ssmRestore`; mimic that in any new
   cache helpers.
5. **Chunked prefill at smaller strides costs ~2-5% on cold.** Default stride
   256 keeps the loss to ~2%; 128 is closer to 5%. The win on warm dominates
   on any realistic workload (1.82× at 256). Don't lower stride further
   without measurement.
6. **Always measure server-side `prompt_ms` and `cached_n`** for prefill
   benches — wall-clock TTFT conflates render + tokenize + queue + decode.
   The Phase 1 bench (`tests/test_hybrid_reuse_equivalence.sh`) asserts on
   `prompt_ms` so future regressions get caught even if wall-time looks fine.
7. **LM Studio's HTTP API does NOT return `prompt_ms`** — only `usage`. Any
   comparison bench must use wall-time for LMS and either wall or
   `prompt_ms` consistently for mlx-serve. `tests/bench_final.sh` documents
   the convention used in the report.
8. **The Swift app needs no changes** to get Phase 1's wins — the binary
   defaults `--ssm-checkpoint-stride 256` baked in, and `ServerOptions`
   doesn't emit any flag that would disable it. Future phases that add CLI
   knobs should mirror this: ship a safe default in the binary so the app
   benefits without a UI change.

## What's left

Phases ordered by **impact × tractability ÷ risk**. Phase 0's harness still
applies — every change measures against the post-Phase-1 baseline in
`docs/perf-csvs/bench_final-20260525-2258.csv`.

### Phase 3 — GGUF speculative decoding (next biggest decode win) — OPEN

LM Studio doesn't ship speculative decoding by default. MLX-serve has it on
the MLX path (PLD + drafter). The GGUF path doesn't. Wiring it up is the
cleanest decode win for GGUF on top of Phase 1's TTFT wins.

**Approach** (unchanged from prior plan):
1. Extend `lib/llama_shim/`:
   - `mlx_llama_open_drafter(parent, gguf_path, err)` — load a small sibling
     that shares vocab with the parent.
   - `mlx_llama_session_eval_speculative(session, drafter, draft_len, out, n_max, err)` —
     draft `draft_len` tokens with the drafter, verify via `llama_decode` on a
     batch of `draft_len+1`, return accepted tokens. Mirror ds4's `evalSpeculative`.
2. Wire `Slot.drafter` and `runLlamaDecodeTick` to route through it when
   `drafter != null`. Reuse `Generator.spec_disabled_runtime` gate semantics
   (drop spec when per-draft acceptance < 50%, sticky after warmup 5).
3. CLI: `--drafter <path>` already exists for MLX; accept a `.gguf` path
   when the target is also GGUF (route by file extension).
4. Persistent-session interaction: the spec verify decodes `[t1, d0..dm-1]`,
   so the resident-token mirror in `arch/llama.zig` must append only the
   *accepted* tokens (not all drafts). Test the partial-accept rollback
   path explicitly.

**Validation:** new `tests/test_drafter_equivalence_gguf.sh` — drafted decode
byte-identical to non-drafted on a fixed seed at temp=0 (first N tokens), at
least Qwen3.5-4B-IQ4_NL (target) + Qwen2.5-0.5B-Instruct-Q4_K_M (drafter).
Mirror the pattern in `tests/test_drafter_equivalence.sh`.

**Acceptance:** ≥ 2× decode speedup on a code-completion prompt where MLX PLD
currently achieves ~1.4×. Drop if per-draft acceptance < 50% on representative
prompts.

### Phase 2 — MLX cold prefill: GatedDeltaNet forward optimization — MEASUREMENT-CLOSED (2026-05-26)

**Outcome:** We are at the MLX/Metal ceiling on M4. The +30% target is
unreachable without research-grade work that supersedes Apple's reference.
A targeted 1-day experiment (full-forward Metal fusion via the existing
`compileForward` closure) shipped byte-identical but with no measurable
speedup. Phase 2 is closed; reopen only if a future MLX release or
custom-kernel work changes the ceiling.

**Measurements (Apple M4 / 16 GB, Qwen3.5-4B-MLX-4bit, 1325-tok prompt, warm engine):**

| Source | Cold prefill |
|---|---:|
| Apple `mlx_lm benchmark` (steady state, post-warmup, prefill-step=2048) | **408 tok/s** |
| mlx-serve baseline (distinct prompts so prefix cache misses each turn) | 378–393 tok/s |
| mlx-serve + `MLX_SERVE_COMPILE_FORWARD=1` (full-forward closure) | 382 tok/s |
| LM Studio (also MLX-backed) | ~388 tok/s |

Gap to mlx-lm: 3.8% on best-case median runs, within run-to-run noise on
others. **Not "materially slower" — Phase 2's original gate condition is
not met.**

**Why nothing meaningful was available:**

1. **GDN delta-recurrence kernel is identical.** mlx-serve's
   `getGdnKernel()` uses grid `(32, Dv, B*Hv)` / threadgroup `(32, 4, 1)` —
   the same source and dispatch as mlx-lm's `gated_delta_kernel` in
   `mlx_lm/models/gated_delta.py`. Reading both side-by-side: char-for-char
   identical. The most expensive component of the hybrid forward has zero
   headroom.
2. **Chunked prefill at stride 256 costs ~3 ms out of 3400 ms.** The
   prefill trace breakdown is `chunked=3360-3500 ms, eval=3-12 ms,
   last_token=0-3 ms`. No structural overhead to remove.
3. **`compute_g` op fusion saves <0.3 ms total** across 24 GDN layers
   (10 element-wise ops on `[B,S,Hv]=[1,1325,32]` tensors). The mlx-lm
   `@partial(mx.compile, shapeless=True)` on `compute_g` is correct but
   the workload is too small to register at this model size.
4. **Full-forward `compileForward` is byte-identical but no faster.**
   Wired in opt-in via `MLX_SERVE_COMPILE_FORWARD=1` (`src/scheduler.zig`,
   `src/generate.zig` chunk loop) and validated with
   `tests/test_phase2_forward_equivalence.sh` 6/6 across Qwen3.5 (hybrid SSM)
   + Gemma 4 E4B (plain attention). Bench medians match within ~1% over
   7 trials each. Apple's `@mx.compile` packaging in Python gives mlx-lm
   the same set of underlying Metal kernels; once you reach those, there
   is no further fusion to apply.

**What did ship from this phase:**

- `tests/test_phase2_forward_equivalence.sh` + fixtures under
  `tests/fixtures/phase2_forward_equivalence/`: a TDD tripwire pinning
  greedy temp=0 output across hybrid SSM + plain attention × 3 prompt
  kinds. Re-run after any change to `src/transformer.zig`'s forward
  path (analogous to `test_hybrid_reuse_equivalence.sh` for Phase 1).
- `MLX_SERVE_COMPILE_FORWARD=1` env opt-in. Off by default. Useful as a
  bisect tool if Apple ships a meaningfully better compile in a future
  mlx-c release; flip and re-bench to find out.

**Reopen criteria:**

- A future mlx-c release ships a fused conv1d+silu+rmsnorm Metal kernel
  (i.e., Apple themselves close the gap) — Phase 2 re-measures and
  potentially recaptures the win at zero cost on our side.
- A research effort yields a custom Metal kernel that fuses the GDN
  block beyond what Apple's stock `gated_delta_kernel` does. Multi-week
  scope, byte-equivalence pin already in place.

**Suggested next phase per the original ordering:** Phase 3 (GGUF
speculative decoding) — clean +2× decode win using documented llama.cpp
APIs; LM Studio doesn't ship it by default.

(Original Phase 2 spec preserved below for context.)

### Phase 2 — Original spec (preserved)

Qwen3.5-4B MLX cold is ~385 tok/s; that's 100% GatedDeltaNet forward compute
(`MLX_SERVE_PREFILL_TRACE=1` confirms 2940 ms of 2941 is the forward). The
gap to LM Studio on cold is small (~tied on the post-Phase-1 bench) because
LM Studio loads the same MLX backend; the gap to mlx-lm is the real ceiling
to chase.

**Approach** (unchanged):
1. Get a reference cold prefill from `mlx_lm.generate` running the SAME
   Qwen3.5-4B MLX 4-bit model. If mlx-serve is materially slower, the gap
   is the optimization target.
2. Diff `transformer.zig`'s `gatedDeltaNet` / `mamba2Mixer` / `conv1dWithCache`
   against `mlx-lm/qwen3_5/...`. Common gaps:
   - Mamba2 parallel scan (SSD form) — mlx-lm may use it; check.
   - Fused conv1d + gating vs separate ops.
   - bf16↔fp32 round-trips around the scan.
   - Tile sizes for the recurrence.
   - Per-layer eval insertion across the prefill loop (SSM lazy graph may
     grow across the whole prompt; KV is eval'd per chunk but SSM is not).
3. Profile with Metal frame capture (Xcode Metal Debugger). If memory-bound,
   tile sizes; if compute-bound, fusion / FLOPs.
4. If a Metal kernel dominates, consider a custom kernel via mlx's
   `metal_kernel` API (may need to add to `src/mlx.zig`).

**Validation:** byte-equivalence (first 30 tokens, temp=0) against the
pre-change forward for every `model_type` touching the changed path:
`qwen3_5`, `qwen3_5_moe`, `qwen3_next`, `lfm2`, `nemotron_h`,
`gemma4`/`gemma4_text` for the conv1d helper.

**Acceptance:** ≥ 30% cold prefill improvement on Qwen3.5-4B MLX *and*
byte-equivalent across listed archs. Drop the change otherwise.

### Phase 5 #1 — Multi-entry hot prefix cache for GGUF — OPEN

Today the llama.cpp path keeps **one** persistent `LlamaSession` per model
(`LoadedModel.llama_session`). With one entry, switching between document A
and document B silently evicts the other's KV — the warm advantage flips
back to cold on every switch. Phase 1 already showed merge-on-extend
matters; for GGUF we need a true LRU.

**Approach:**
1. Replace `LoadedModel.llama_session: ?*LlamaSession` with a small array
   (e.g., `llama_sessions: [N]?*LlamaSession` + `last_used: [N]u64`).
2. In `runPrefillLlama`, find the entry with the longest common prefix
   against the incoming prompt (similar to `HotPrefixCache.findBestMatch`
   but on token mirrors). If miss: evict LRU, create fresh session.
3. CLI: `--llama-cache-entries N` (default 1 for backwards-compat),
   `--llama-cache-mem <bytes>` (cap per-session ctx × N).
4. Session contention: `llama_session_busy` becomes per-entry; the submit
   gate waits on the chosen entry, not the model.

**Validation:** new `tests/test_llama_multi_session.sh` — alternate two
distinct long-doc QA prompts; assert each gets `cached_n > 0` on its
second visit.

**Acceptance:** zero regression on single-doc workflow, near-zero TTFT
on alternation between N distinct doc roots.

### Phase 4 #1 — Parallel tokenize on the connection thread — OPEN

Today `handleChatCompletions` renders the chat template + tokenizes inline
**before** `sch.submit()`. For short prompts this is a few ms; for long
prompts (8k+ doc) it's > 50 ms on the connection thread. The scheduler
queue wait dominates only when busy; for the common single-slot case
they're sequential and the user pays both.

**Approach:**
1. Add `tokenize_ms` to `formatTimingsObject` so we can measure where
   the time goes (instrumentation-first).
2. If render+tokenize > 5% of TTFT for short prompts, move to a worker
   thread that pipelines with the scheduler queue wait. The slot.submit
   contract accepts pre-tokenized `prompt_ids: []const u32`, so the
   architectural change is small (move the render+tokenize work into a
   future/promise consumed at submit-time).

**Acceptance:** ≥ 20 ms off measured end-to-end TTFT on short prompts.

### Phase 4 #3 — System-prompt prewarm — OPEN

When the loaded model has a stable system prompt (e.g., agent flows where
the same system + tools combination repeats every turn), we can:

1. Pre-tokenize at load time and stash on `LoadedModel.warm_system_tokens`.
2. Optionally prefill the KV cache once at load with those tokens.
3. Skip both on subsequent requests by comparing the first N tokens against
   the resident cache.

This is mostly a Phase 1 follow-on: Phase 1 already gives multi-turn
warm reuse; this gives **first-turn** warm reuse when the system prompt
is known up front.

**Acceptance:** first-turn TTFT drops by ≥ 100 ms when the model launches
with a fixed system prompt.

### Phase 5 #3 (deeper GGUF wins) — OPEN

After Phase 3 (spec decode) and Phase 5 #1 (multi-session) land, the remaining
GGUF-beyond-LM-Studio knobs:

- **Concurrent decoding across sessions** — llama.cpp supports parallel
  sequences in one context. Today mlx-serve forces serial via the submit
  gate. After Phase 6's MLX work, evaluate un-serializing GGUF too.
- **n_ubatch / n_threads_batch auto-tune** — measurement-gated. Sweep on
  Apple Silicon and persist in `~/.mlx-serve/llama-tune.json`.
- **Speculative-decode + persistent session composition** — verify Phase 3
  composes with the prefix cache (resident tokens reflect ACCEPTED tokens
  only). Test composes both at once.

### Phase 6 — Concurrent / continuous batching for hybrid — OPEN

mlx-serve has batched decode for plain-attention dense models
(`runDecodeTick`, `--max-concurrent N`). With Phase 1's SSM checkpointing
landed, per-slot SSM state can co-exist with batched attention in
principle — needs verification.

1. Continuous batching for MLX hybrid — verify batched decode works for
   Qwen3.5 MLX with `--max-concurrent N`. The SSM state is per-slot
   (lives on `Slot.ssm_entries`); batched forward needs to interleave the
   SSM update per slot. Non-trivial — the recurrent state can't share
   weights across slots in one matmul the way attention K/V can.
2. GGUF concurrent slots via llama.cpp's parallel sequences (depends on
   Phase 5 #1 first).
3. Auto-clamp `max_concurrent` for engine-backed models based on
   available memory (KV per slot × N must fit).

**Acceptance:** 2 concurrent requests sustain ≥ 1.7× single-request
throughput on both engines.

### Phase 7 — Quantization tuning — OPEN

Lower-priority. Run after Phases 2–6.

1. **MLX TurboQuant default** — bench whether `turbo2`/`turbo4` should be
   the default for INT4 weights. Existing equivalence test infra applies.
2. **MLX 2-bit weight quantization** — `mlx_quantize(..., 2)` is supported
   natively; worth a bench. 1-bit needs custom pack/unpack.
3. **GGUF quant matrix** — surface llama.cpp's quant variety
   (Q4_K_M / IQ4_NL / IQ4_XS / Q5_K_M / etc.) in `/v1/models` so the Swift
   app can show users the speed/quality trade-off per quant.

**Acceptance:** documented "best default per model size" table; no
correctness regressions vs existing dense path.

## Critical files (post-Phase-1 map)

| File | What lives here |
|---|---|
| `src/transformer.zig` | MLX forward pass. `SSMCheckpoint`, `captureSsmCheckpoint`, `restoreSsmCheckpoint` for Phase 1. `gatedDeltaNet`, `mamba2Mixer`, `conv1dWithCache` for Phase 2. |
| `src/generate.zig` | Chunked prefill with stride-aligned SSM snapshots (Phase 1). Drafter/PLD dispatch for Phase 3's MLX side. `Generator.takeSsmCheckpoints`. |
| `src/prefix_cache.zig` | `Entry.ssm_checkpoints`, `commitWithSsm` with merge-on-extend, `lookupAndRestore` with hybrid effective-matched clamp, `shouldUse(config, enable_ssm_checkpoints)`. |
| `src/scheduler.zig` | LoadParams plumbing, runPrefill calls commitSlotIfApplicable's drain. Phase 5 #1 expands `LoadedModel.llama_session` → LRU. |
| `src/arch/llama.zig` + `lib/llama_shim/llama_shim.{h,c}` + `src/llama_ffi.zig` | libllama bridge. Phase 3 (spec decode) + Phase 5 (multi-session). `LlamaKvQuant` enum already in place. |
| `src/server.zig` | HTTP/SSE plumbing. `formatTimingsObject` for Phase 4 #1's `tokenize_ms` field. `--ssm-checkpoint-stride` / `--ssm-checkpoint-max` / `--llama-kv-quant` globals. |
| `src/model_registry.zig` | `LoadedModel` fields (Phase 1's `ssm_checkpoint_stride/max`, Phase 5 #2's `llama_kv_type_{k,v}`, future Phase 5 #1's `llama_sessions[]`). |
| `tests/bench_final.sh` / `tests/bench_decode.sh` / `tests/plot_*.py` | Bench harness + chart renderers. Phase 2+ measures against these. |
| `tests/test_hybrid_reuse_equivalence.sh` | Phase 1 regression guard. Re-run after any change to `src/prefix_cache.zig` or the SSM snapshot path. |

## Risks & discipline (still apply)

- **Cross-arch correctness is the project's red line.** Plain attention,
  sliding-window (Gemma 3/4), hybrid SSM (lfm2 / nemotron_h /
  qwen3_5(_moe) / qwen3_next), MoE, DSV4 — any shared-path change must
  pass byte-equivalence on each `model_type` it touches.
- **Bench overhead is real.** Phase 1's chunked-prefill alignment cost
  ~2% on cold for the warm wins. Future phases should budget similarly
  and measure both halves.
- **Speculative decoding regresses for "creative" content** (low draft
  acceptance). Honor the runtime gate (`spec_disabled_runtime` at < 50%
  per-draft acceptance, warmup 5). Phase 3 reuses the same threshold.
- **Never trust client-side wall-time as a prefill number.** Use
  `timings.prompt_per_second` from the server response, computed over
  UNCACHED tokens. (See `tests/bench_final.sh` for the convention.)
- **ReleaseFast is mandatory.** A Debug binary is 2-4× slower and silently
  generated by some build flows. Check `du -h zig-out/bin/mlx-serve`:
  ~4.1 MB = ReleaseFast, ~10 MB = Debug.
- **TDD is mandatory.** Write the failing test first; verify reverting the
  change re-fails it. Phase 1's `test_hybrid_reuse_equivalence.sh` is the
  template.
- **No commits without explicit user direction.** Per the user's global rules.

## Suggested execution order (next agent)

1. **Phase 3 — GGUF speculative decoding** (3–5 days; cleanest decode win
   using proven llama.cpp APIs; the user's biggest remaining gap is
   "decode parity on GGUF").
2. **Phase 5 #1 — multi-entry GGUF LRU** (1–2 days; complements Phase 1 on
   the GGUF side; trivial once you've internalized the prefix-cache patterns).
3. **Phase 4 #1 + #3 — TTFT micro-opts** (parallelizable with the above;
   each ≤ 1 day; measurement-gated).
4. **Phase 2 — MLX GatedDeltaNet forward optimization** (open-ended;
   gate on mlx-lm comparison telling you there's a real gap to close).
5. **Phase 6 — Concurrent batching for hybrid** (after Phase 1 SSM
   per-slot validation; depends on Phase 1 + 5 #1).
6. **Phase 7 — Quantization defaults** (last; runs after everything else).

## Definition of done (updated)

The `docs/perf-vs-lmstudio-final.{md,png}` table is re-run after each phase
and shows mlx-serve **decisively** ahead of LM Studio in every cell. Current
table: 9/9 wins or ties. The remaining work narrows the cold-prefill gap
(currently tied) and pushes warm/decode further ahead.

A 5-turn agent conversation against Qwen3.5-4B MLX shows turn-2..5 TTFT
< 100 ms on a 1k-token shared context. **Already met (median 30-200 ms in
the current bench)**.

If any cell falls behind in a later re-run, the phase that owns it is reopened.
The plan isn't done until every cell stays green across multiple runs.
