# Performance plan — decisively beat LM Studio in every category

## Goal

Beat LM Studio by a measurable, decisive margin across every dimension that
affects user-perceived speed, on **both** engines mlx-serve ships:

| Category | GGUF (via llama.cpp) | MLX (safetensors) |
|---|---|---|
| Cold prefill (tok/s) | beat by ≥ 25% | beat by ≥ 25% |
| Warm / multi-turn prefill (TTFT) | tied/better — both near-instant | tied/better — both near-instant |
| Decode (tok/s) | beat by ≥ 30% | beat by ≥ 30% |
| End-to-end TTFT, 1k prompt cold | beat by ≥ 30% | beat by ≥ 30% |
| End-to-end TTFT, warm multi-turn | beat by ≥ 5× | beat by ≥ 5× |

"Beat LM Studio" is harder than it sounds for GGUF because we share libllama
with them — the wins must come from things LM Studio doesn't do (speculative
decoding, multi-entry prompt caching, tokenization pipelining, concurrent
batching). For MLX we own the entire stack, so the ceiling is higher but the
work is deeper (GatedDeltaNet forward, SSM checkpointing).

## Where we are (measured, post the prefill/TTFT work)

Run `./tests/bench_prefill.sh <model>` to reproduce. Numbers below are
Qwen3.5-4B on Apple Silicon, 750-token prompt, ReleaseFast.

| | Cold prefill | Multi-turn (warm) | Decode |
|---|---|---|---|
| **GGUF Qwen3.5-4B-IQ4_NL** | ~698 tok/s | reuses 940/941 → near-instant | ~parity with LM Studio |
| **MLX Qwen3.5-4B-4bit** | ~383 tok/s | **no reuse — full cold every turn** | ~8% behind LM Studio |

Key facts the next agent **must** internalize before changing anything:

1. **The original "384 tok/s for both engines" complaint was a client-TTFT
   artifact, not raw prefill compute.** GGUF cold is actually ~700 tok/s.
   Always measure server-side `timings.prompt_per_second` and `timings.cached_n`
   (Phase 1 of the prior work surfaced these in every API response).
2. **GGUF multi-turn is solved** — a persistent `LlamaSession` per model
   reuses the KV prefix (LM-Studio-style). Don't redo this.
3. **MLX hybrid (GatedDeltaNet) reuse is architecturally blocked** with the
   current SSM cache design. The prior session attempted it and reverted; see
   memory `project_prefill_perf_findings`. The real fix needs **per-position
   SSM checkpointing** (Phase 2 below) — this is the highest-impact unblocker.
4. **MLX cold prefill (383 tok/s)** is 100% GatedDeltaNet forward compute
   (`MLX_SERVE_PREFILL_TRACE=1` shows 2940 ms forward, 1 ms eval — chunk size
   has zero effect). Optimizing requires touching `transformer.zig`.
5. **Cross-arch byte-equivalence is non-negotiable.** Plain attention, hybrid
   SSM (lfm2 / nemotron_h / qwen3_5(_moe) / qwen3_next), sliding window (Gemma
   3/4), MoE — touching any shared path requires validating each.
6. **Always build `-Doptimize=ReleaseFast`.** Plain `swift build` (without
   `app/build.sh`) silently rebuilds zig-out in Debug.

## Approach

The phases are ordered by **impact × tractability ÷ risk**. Phase 0 is mandatory
before any compute change. Phase 1 unlocks the MLX multi-turn win that the
prior work surfaced as the user's biggest pain. Each phase has its own
acceptance criteria; if the measured gain doesn't clear the bar, the phase is
dropped (per the project's "no speculative changes" discipline).

### Phase 0 — Apples-to-apples baseline harness (mandatory)

Without this every later phase is fiction. Extend `tests/bench_prefill.sh` and
add comparative variants:

- `tests/bench_vs_lmstudio.sh` — same prompt, same params, three back-ends:
  mlx-serve (HTTP), LM Studio (HTTP, port 1234), mlx-lm (subprocess) — captures
  cold prefill / warm prefill / decode tok/s / first-byte TTFT / total wall time
  per scenario. Format CSV + a markdown summary table.
- Scenarios (per model+engine pair):
  1. **Single-shot cold** — fresh server, unique 1k-token prompt, max_tokens=64.
  2. **Multi-turn warm** — 5-turn conversation, each turn ~200 tokens of new
     user text, max_tokens=64. Measure turn 2..5 TTFT.
  3. **Long-doc QA** — 8k-token document + short question. Cold and warm
     (same doc, different question second time).
  4. **Decode-dominated** — short prompt, max_tokens=512. Pure decode rate.
- Models to bench (need to be downloaded):
  - GGUF: `Qwen3.5-4B-IQ4_NL`, `Qwen3.5-4B-Q4_K_M`, `gemma-4-E4B-it-Q4_K_M`,
    `Llama-3.1-8B-Instruct-Q4_K_M`.
  - MLX: `Qwen3.5-4B-MLX-4bit`, `gemma-4-E4B-it-4bit` (plain-attention
    reference — critical for isolating the GatedDeltaNet cost),
    `Llama-3.1-8B-Instruct-4bit-MLX`.
- **Output**: a markdown table per model showing the GAP vs LM Studio per
  scenario. Commit it as `bench/baseline-{date}.md`. Every subsequent phase
  compares against this baseline.

Acceptance: the table exists, reproduces, and clearly identifies which numbers
need to move. Without it, stop.

### Phase 1 — MLX hybrid prefix reuse via per-position SSM checkpointing

**The biggest single MLX TTFT win.** Today every multi-turn MLX request pays
full cold prefill because the hot prefix cache is disabled for hybrid
(`has_hybrid_layers || full_attention_interval > 0`) — the SSM/recurrent state
isn't positionally indexable, so it can't be trimmed to an arbitrary prefix.

The prior session attempted snapshotting at one boundary (post-generation,
then prompt-boundary) and proved both fail due to **BPE boundary shift**:
turn 1's prompt tokenizes to 159 tokens, turn 2 shares only 155 (the trailing
assistant-header tokens re-tokenize differently when content follows). The
snapshot was at 159 but the usable reuse point is at 155 → can't reuse.

**The real fix is what llama.cpp does**: keep multiple SSM checkpoints along
the sequence, not just one. (llama.cpp exposes this as `n_rs_seq` — number of
recurrent-state snapshots per sequence.)

**Approach**:

1. Extend `SSMCacheEntry` (or add `SSMCheckpointBuffer`) to hold an array of
   SSM-state snapshots at positions `0, K, 2K, 3K, ...` where `K` is the
   checkpoint stride (e.g., 128 tokens). For Qwen3.5 the SSM state per layer
   is `[B, Hv, Dv, Dk]` — a few MB; M checkpoints × N layers × that = bounded.
2. During prefill / decode, every time the cache position crosses a multiple
   of `K`, snapshot the current SSM state into the checkpoint buffer.
3. In `prefix_cache.commit`, capture the full checkpoint array alongside the
   KV snapshot.
4. In `prefix_cache.lookupAndRestore`, when matched length `m` < entry length,
   find the largest checkpoint position `p ≤ m`, restore SSM from that
   checkpoint, then re-forward the small tail `tokens[p..m]` through the
   GatedDeltaNet/SSM layers to advance state to position `m`. KV is already
   restored to `m` via the existing positional KV restore (truncate).
5. Cap memory: `--ssm-checkpoint-stride N` and `--ssm-checkpoint-budget` flags.
   Default stride 128 (so the re-forward tail is at most 127 tokens, fast).
6. Re-enable hybrid in `HotPrefixCache.shouldUse` (precise gate that does NOT
   regress plain attention — see the prior session's notes on `has_sliding_window`
   defaulting to true; the right gate is `!(has_sliding_window AND
   full_attention_interval > 0)`).

**Validation (TDD)**:
- Unit test for the checkpoint stride math (which checkpoint covers position
  `m`, how many tail tokens to re-forward).
- New shell test `tests/test_hybrid_reuse_equivalence.sh` — multi-turn warm
  output byte-identical to cold for first N tokens at temp=0, AND `cached_n`
  on turn 2..5 > a meaningful threshold (e.g., ≥ 80% of turn-1 prompt length).
- Cross-arch: gate on `MLX_HYBRID_MODEL` env (default Qwen3.5-4B), but make the
  test runnable on `qwen3_5_moe`, `qwen3_next`, `lfm2`, `nemotron_h` and run it
  on each before merging. Use `tests/test_pld_equivalence.sh` as the structural
  template — same first-N-tokens byte-equivalence pattern.

**Acceptance**: multi-turn TTFT collapses to suffix-prefill cost on Qwen3.5
MLX (turn 2..5 prefill compute < 20% of turn-1's). Byte-identical to cold. No
plain-attention regression.

**Risk gate**: if cross-arch correctness can't be established for any single
listed model, ship the feature behind a per-arch allowlist rather than a
global flip. Document.

### Phase 2 — MLX cold prefill: GatedDeltaNet forward optimization

The 383 tok/s on Qwen3.5-4B MLX is 100% forward compute (`MLX_SERVE_PREFILL_TRACE=1`
confirms 2940 ms of 2941 is the forward; chunked-eval overhead is 1 ms).
Chunk size has no effect (256→8192 all ~2920 ms). So the bottleneck is the
GatedDeltaNet implementation in `transformer.zig`.

**Approach**:

1. Get a reference cold prefill number from mlx-lm running the SAME Qwen3.5-4B
   MLX 4-bit model — that's the realistic ceiling on this hardware. If
   mlx-serve is materially slower, the gap is the optimization target.
2. Diff `mlx-serve`'s `gatedDeltaNet` / `mamba2Mixer` / `conv1dWithCache`
   against the latest `mlx-lm/qwen3_5/...` (or `mlx-lm/qwen3_next/...`).
   Common gaps to look for:
   - **Parallel scan** — Mamba2/GatedDeltaNet's recurrence has a chunked
     parallel form (the "Mamba2 SSD" formulation). mlx-lm may use it; check.
   - **Fused conv1d + gating** — separate ops vs one fused call.
   - **Redundant casts / copies** — bf16↔fp32 round-trips around the scan.
   - **Tile sizes** — head_dim chunking, sequence chunking inside the recurrence.
   - **Per-layer eval insertion** — the prefill loop in `generate.zig` evals
     KV per chunk but **not SSM**; the SSM lazy graph may grow across the whole
     prompt, hurting cache locality.
3. Profile with Metal frame capture (Xcode's Metal Debugger) — identify which
   kernel dominates and whether it's memory-bound or compute-bound. If
   memory-bound, the win is in tile sizes; if compute-bound, the win is in
   reducing FLOPs / fusion.
4. If a Metal kernel is the bottleneck and MLX's high-level op set is
   suboptimal, consider a custom Metal kernel via mlx's `metal_kernel` API
   (mlx-c may not expose this — possible add to `mlx.zig`).

**Validation**: byte-equivalence (first 30 tokens, temp=0) against the
pre-change forward, for every model_type that touches the changed code path
(`qwen3_5`, `qwen3_5_moe`, `qwen3_next`, `lfm2`, `nemotron_h`,
`gemma4`/`gemma4_text` for the conv1d helper, `deepseek_v4` if shared).

**Acceptance**: ≥ 30% cold prefill improvement on Qwen3.5-4B MLX *and*
byte-equivalent across the listed archs. Drop the change otherwise.

### Phase 3 — Decode parity: speculative decoding for GGUF

LM Studio doesn't ship speculative decoding by default. MLX already has it
(PLD + drafter, `tests/test_pld_equivalence.sh`, `tests/test_drafter_equivalence.sh`).
The GGUF path doesn't. Wiring it up — using llama.cpp's existing batch decode
API — is the cleanest decode win for GGUF.

**Approach**:

1. Add draft-model loading to `lib/llama_shim/`:
   - `mlx_llama_open_drafter(parent_engine, gguf_path, err)` → loads a small
     sibling model that shares vocab with the parent.
   - `mlx_llama_session_eval_speculative(session, drafter, draft_len, out_tokens,
     n_max, err)` → drafts `draft_len` tokens with the drafter, verifies with
     the target via `llama_decode` on a batch of `draft_len+1`, returns
     accepted tokens. Mirror the ds4 `evalSpeculative` shape so the scheduler
     dispatch is uniform.
2. Wire `slot.drafter` and `runLlamaDecodeTick` to route through it when
   `drafter != null`. Reuse `Generator.spec_disabled_runtime` gate semantics
   (drop spec when per-draft acceptance < 50%).
3. CLI: `--drafter` already exists for MLX; accept a `.gguf` path when the
   target is also GGUF.
4. Persistent-session interaction: the spec verify decodes `[t1, d0..dm-1]`,
   so the resident-token mirror in `arch/llama.zig` must append only the
   *accepted* tokens (not all drafts). Test the partial-accept rollback
   path explicitly.

**Validation**: new `tests/test_drafter_equivalence_gguf.sh` — drafted decode
byte-identical to non-drafted on a fixed seed at temp=0 (first N tokens), for
at least Qwen3.5-4B-IQ4_NL (target) + Qwen2.5-0.5B-Instruct-Q4_K_M (drafter).
Mirror the pattern in `tests/test_drafter_equivalence.sh`.

**Acceptance**: ≥ 2× decode speedup on a creative-writing prompt where MLX
PLD currently achieves ~1.4× (same workload class). Drop if per-draft
acceptance is < 50% on representative prompts.

### Phase 4 — TTFT-specific micro-optimizations (orthogonal to compute)

Once Phases 1–3 land, TTFT is dominated by everything-but-compute. Measure
where the time goes (Phase 0 harness should already split this).

**Likely wins**:

1. **Jinja render + BPE encode on the connection thread, in parallel with
   scheduler queue wait.** Today `handleChatCompletions` renders + tokenizes
   inline before `submit`. Profile; if it's > 5% of TTFT for short prompts,
   move tokenization off the critical path.
2. **First SSE chunk emitted before any decode** — the `role`/`created` chunk
   can fire on slot admission, not after the first sampled token. Knocks
   ~10–50 ms off perceived TTFT for streaming clients.
3. **System-prompt pre-warm**: if the loaded model has a chat template and a
   stable system prompt, pre-compute its tokenization + (optionally) prefill
   it into the persistent session at load time. Subsequent requests skip both.
4. **HTTP/2 + persistent connections** — `Conn` is currently one-shot per
   request. Multi-turn agents (Claude Code, Cursor) re-handshake every call.

**Acceptance**: each item lands only if it shaves ≥ 20 ms off measured
end-to-end TTFT (Phase 0 harness). Stack-rank by measured impact.

### Phase 5 — GGUF: go beyond LM Studio

The persistent session in Phase 3 already matches LM Studio's prompt cache.
To **beat** it on GGUF (same library underneath), do things LM Studio doesn't:

1. **Multi-entry hot prefix cache for GGUF too.** Today the llama path keeps
   ONE persistent session (capacity 1). Add an LRU of N sessions per model
   so doc-A and doc-B requests don't evict each other. (Memory cost: N × KV.
   Cap via `--llama-cache-entries` and `--llama-cache-mem`.)
2. **Per-layer KV quantization for GGUF** — llama.cpp exposes `type_k` /
   `type_v` in `llama_context_params`. Expose `--llama-kv-quant 4|8` in
   mlx-serve. Half/quarter the KV memory → more concurrent slots.
3. **Concurrent decoding across sessions** — llama.cpp supports parallel
   sequences in one context. Today mlx-serve forces serial via the submit
   gate. After Phase 1 lands an analogous mechanism for MLX, evaluate
   un-serializing GGUF for the parallel case.
4. **n_ubatch / n_threads_batch auto-tune** — measurement-gated. The shim
   currently uses libllama defaults. Sweep on Apple Silicon and pick the
   best per (model size × quant) tuple, persist in `~/.mlx-serve/llama-tune.json`.
5. **Speculative-decode + persistent session interaction** — verify Phase 3
   composes with the prefix cache (resident tokens must reflect ACCEPTED
   tokens only). Test composes both at once.

**Acceptance**: ≥ 25% cold prefill improvement on GGUF (vs LM Studio same
quant/model) OR ≥ 20% decode improvement, with no correctness regression.

### Phase 6 — Concurrent / continuous batching

mlx-serve has a batched-decode path for MLX (`runDecodeTick`), gated to plain
attention. Bring it to feature parity:

1. **Continuous batching for MLX hybrid** — once SSM checkpointing (Phase 1)
   lands, per-slot SSM state can co-exist with batched attention. Verify
   batched decode works for Qwen3.5 MLX with `--max-concurrent N`.
2. **GGUF concurrent slots via llama.cpp's parallel sequences** — see Phase 5
   #3. Requires the shim to expose `llama_n_seq_max` + `seq_id` per decode.
3. **Auto-clamp `max_concurrent`** for engine-backed models based on
   available memory (KV per slot × N must fit). Today it's a manual flag.

**Acceptance**: 2 concurrent requests sustain ≥ 1.7× single-request
throughput (per-slot) on both engines. Existing `tests/bench_concurrent.py`
already measures this — extend to GGUF.

### Phase 7 — Quantization tuning

Lower-priority, high-leverage. Run after Phases 0–4.

1. **MLX TurboQuant default** — already shipped; bench whether
   `turbo2`/`turbo4` should be the default for INT4 weights (the prior work
   noted byte-stable past 30 tokens at temp ≥ 0.01; verify on Qwen3.5).
2. **MLX 2-bit weight quantization** — the CLAUDE.md notes 1-bit needs custom
   pack/unpack; 2-bit may be tractable via `mlx_quantize(..., 2)`. Worth a bench.
3. **GGUF quant matrix** — surface llama.cpp's quant variety
   (Q4_K_M / IQ4_NL / IQ4_XS / Q5_K_M / etc.) in `/v1/models` so the Swift app
   can show users the speed/quality trade-off per quant.

**Acceptance**: a documented "best default per model size" table; no
correctness regressions vs existing dense path.

## Critical files

| File | What lives here |
|---|---|
| `src/transformer.zig` | MLX forward pass (`gatedDeltaNet`, `mamba2Mixer`, `conv1dWithCache`, `KVCache`, `SSMCacheEntry*`, `KVCacheSnapshot`, `ssmSnapshot`/`ssmRestore`). Phase 1, Phase 2. |
| `src/prefix_cache.zig` | Hot prefix cache. Phase 1 extends `Entry` with SSM checkpoint arrays; `shouldUse` gates per arch. |
| `src/scheduler.zig` | Slot lifecycle, prefill/decode dispatch, `commitSlotIfApplicable`, persistent llama session gate. Phases 1, 3, 5, 6. |
| `src/generate.zig` | Chunked prefill loop, decode, PLD/drafter dispatch, `Generator.initWithOptions`. Phase 2 (per-chunk SSM eval), Phase 3 (spec-decode dispatch for llama). |
| `src/arch/llama.zig` + `src/llama_ffi.zig` + `lib/llama_shim/llama_shim.{h,c}` | libllama bridge. Phase 3 (spec decode), Phase 5 (multi-session, KV quant). |
| `src/server.zig` | HTTP / SSE plumbing, `handleChatCompletions`, timing reporting (`formatTimingsObject`, `formatPerfBracket`). Phase 4 (tokenize-on-conn-thread, early SSE chunk). |
| `tests/bench_prefill.sh` | Existing cold/warm harness — extend for Phase 0. |
| `bench.sh` / `tests/bench_*.{sh,py}` | Existing bench infra; pattern for new comparative harnesses. |
| `tests/test_pld_equivalence.sh` / `tests/test_drafter_equivalence.sh` | Byte-equivalence test templates — mirror for Phase 1 and Phase 3 GGUF spec decode. |

## Reuse / patterns

- **SSM snapshot/restore** (`transformer_mod.ssmSnapshot`, `ssmRestore`,
  `ssmSnapshotDeinit`) — proven by PLD for intra-request rollback across
  every hybrid arch. Phase 1 reuses these for cross-request checkpoints.
- **Persistent engine session pattern** — `LoadedModel.llama_session` +
  `llama_session_busy` gate in `Scheduler.submit`/`complete`. The MLX
  equivalent (Phase 1, Phase 6) follows the same shape.
- **Byte-equivalence test infra** — `tests/test_pld_equivalence.sh` (first-N
  tokens at temp=0, env-gated on a model fixture). Every correctness-critical
  phase ships a test in this shape.
- **Server timing JSON** (`formatTimingsObject`) — already exposes
  `prompt_n`, `cached_n`, `prompt_per_second`, `predicted_per_second`. New
  fields the next agent will want:
  - `tokenize_ms` (Phase 4 #1 — measure Jinja+BPE cost).
  - `first_chunk_ms` (Phase 4 #2 — TTFT excluding decode).
  - `accepted_drafts` / `proposed_drafts` (Phase 3 — speculative-decode telemetry).
- **Decision-gate framing** — the prior `prefill/TTFT` plan's "drop the
  sub-cause if Phase 2 evidence doesn't support it" is the right discipline.
  Carry it forward: no speculative changes, every phase has measured
  acceptance, drop sub-causes that miss the bar.

## Risks & discipline

- **Cross-arch correctness is the project's red line.** Plain attention,
  sliding-window (Gemma 3/4), hybrid SSM (lfm2 / nemotron_h / qwen3_5(_moe) /
  qwen3_next), MoE, DSV4 — any shared-path change must pass byte-equivalence
  on each model_type it touches. Don't ship single-arch wins as global flips.
- **Per-position SSM checkpoints have a memory cost.** Bound them. Phase 1's
  default stride should keep total checkpoint memory < the KV cache size
  itself; bench memory before-and-after.
- **Speculative decoding regresses for "creative" content (low draft
  acceptance).** Honor the runtime gate (`spec_disabled_runtime` at < 50%
  per-draft acceptance, warmup 5 ticks — see CLAUDE.md). Phase 3 reuses the
  same threshold/warmup.
- **The "384 for both engines" measurement was wrong.** Never trust
  client-side TTFT as a prefill number. Use `timings.prompt_per_second` from
  the server response, computed over UNCACHED tokens (Phase 1 of the prior
  work).
- **ReleaseFast is mandatory.** A Debug binary is 2-4× slower and silently
  generated by some build flows (plain `swift build` triggers a Debug zig
  rebuild). Check `du -h zig-out/bin/mlx-serve` after any build: ~4.3 MB =
  ReleaseFast, ~10 MB = Debug — rebuild ReleaseFast or use `app/build.sh`.
- **TDD is mandatory** for every code change (per CLAUDE.md). Write the
  failing test first; verify reverting the change re-fails it.
- **No commits without explicit user direction.** Per the user's global rules.

## Suggested execution order

1. Phase 0 — baseline harness + comparative numbers (1–2 days).
2. Phase 1 — MLX hybrid prefix reuse via SSM checkpointing (4–7 days; the big
   unlock for MLX TTFT).
3. Phase 3 — GGUF speculative decoding (3–5 days; the cleanest decode win,
   uses proven llama.cpp APIs).
4. Phase 2 — MLX GatedDeltaNet forward optimization (open-ended; gate on
   Phase 0's mlx-lm comparison telling you there's a real gap to close).
5. Phase 4 — TTFT micro-optimizations (parallelizable with the above).
6. Phase 5 — GGUF-beyond-LM-Studio polish (after Phase 3 lands).
7. Phase 6 — Concurrent batching (after Phase 1 unblocks MLX hybrid).
8. Phase 7 — Quantization defaults (last; runs after everything else).

## Definition of done

The `bench/baseline-{date}.md` table from Phase 0 is re-run and shows
mlx-serve **decisively** ahead of LM Studio in every cell, on every benched
(model, engine, scenario) combination. The `tests/test_*.sh` suite is green.
A 5-turn agent conversation against Qwen3.5-4B (either engine) shows
turn-2..5 TTFT < 100 ms on a 1k-token shared context.

If any cell still shows mlx-serve behind, the phase that owns it is reopened.
The plan isn't done until the whole table is won.
