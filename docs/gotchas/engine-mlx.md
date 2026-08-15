# Engine: KV cache, spec-decode, kernels, MLX/FFI — war stories (moved out of CLAUDE.md)

Full histories: live failures, measurements, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### KV cache after tool calls
Generated tool-call tokens are in the cache but not in `cached_prompt_ids` → reusing for the next request (with tool results) corrupts attention. Auto-invalidated. Pad-only generations also trigger invalidation.

### Sliding window KV cache
Gemma 4 E4B (512-token window) keeps full BUFFER — nothing is ever dropped. What is trimmed is the VIEW handed to attention: `slidingViewFor` sizes it to the tail this forward can reach. Matches mlx-lm `RotatingKVCache` on the buffer; see "The sliding trim was decode-only for every arch" below for the view.

### Hot-cache restore must CLAMP the cache offset to the matched length, never trust the snapshot length (offset-drift / mask-crash class)
The `offset` passed to a forward drives BOTH the RoPE positions (`mlx_fast_rope(..., offset, ...)`) and the attention mask width (`total_kv = offset + seq_len`). The KV buffer that mask attends to has length `entry.offset + new_len` (the CACHE's own offset). These two offsets MUST stay equal — if they drift, RoPE positions are wrong AND (on Gemma sliding-window layers, which build an explicit `"array"` mask via `createSlidingWindowMask(seq_len, total_kv, sw)`) the mask can't broadcast against the KV, crashing the whole process with `MLX error: [broadcast_shapes] Shapes (1,1,Q,total_kv) and (1,H,Q,buf_len) cannot be broadcast`. Other archs take SDPA's `"causal"` fast path (no explicit mask) so they only suffer the silent RoPE drift, not the crash.
- Live 2026-07-09 (soak, gemma-4-26B-A4B at ~16K ctx, multi-turn agent): mask 16890 vs KV 16892 — a 2-token drift. ROOT: PLD/speculative decode leaves STALE draft positions in the KV buffer past the committed step (the buffer offset advances during a verify forward and the partial-accept rollback doesn't shrink it below the committed count), and `HotPrefixCache.commit` snapshots the inflated buffer — so the entry's KV snapshot is LONGER than its logical `tokens.len`. On the next request that matches the entry's ENTIRE token sequence but is longer (a PARTIAL hit: `effective_matched == e.tokens.len < prompt_ids.len`), the restore's truncate was guarded by `if (final_len < e.tokens.len)` — FALSE in exactly this case — so the restored `cache.offset` kept the inflated snapshot length while generation tracked `moe_seq_offset = effective_matched`. Drift.
- Fix (`prefix_cache.zig` `lookupAndRestore`): ALWAYS `truncate(final_len)` on restore, never guard it. `truncate` is a no-op when the buffer is already `final_len`, so unconditional clamping is safe and restores the invariant `cache.offset == matched`. The stale KV tail has no matching token id (the match runs against `e.tokens`), so it can never be reached — discarding it is correct. Symptom signature: a `broadcast_shapes` crash whose two shapes differ ONLY in the last (key) dim by a small amount, after a `[hot-cache] reused N/M` line where N equals the committed entry's full length. Rule: on ANY cache restore, the post-restore offset is the MATCHED length, not the snapshot's stored length — clamp it. Guarded by the `restore clamps an inflated snapshot to the matched length` test in prefix_cache.zig (red-on-revert: `expected 64, found 66`). NOTE: the deeper root (PLD leaving a stale buffer tail past the committed step) is DEFENDED here but not eliminated at source — the restore clamp makes it harmless; a belt-and-braces trim at `commit` time would also be valid.

### Lazy mlx_slice views fed to gather_qmm (silent wrong/zero expert outputs)
A weight produced by `mlx_slice` is a lazy VIEW with parent strides. Feeding such a view (weight, scales, or biases) to `mlx_gather_qmm` SILENTLY computes wrong expert outputs at real-checkpoint scale — observed as exactly-zero MoE branch output on DiffusionGemma's split `experts.gate_up_proj` (the model still "worked", just incoherently: the dense shared-MLP branch carried it). Three traps make this class nasty:
- Reading the slice via data pointers materializes it correctly, so value-equality tests on the split pass while the gather is broken — the assertion must go THROUGH gather_qmm.
- Toy/synthetic geometries stay green: a full-geometry random-weight repro (E=128, M=704, IN=2816, sorted path) showed ZERO diff between view and contiguous. Only the real checkpoint reproduces it. The guard is therefore the LIVE converged-canvas self-consistency test (`DIFFUSION_TEST_MODEL`), verified red-on-revert (49/64 vs 63/64).
- Rule: any weight that goes through `mlx_gather_qmm`/`mlx_quantized_matmul` and was born from a slice must be materialized with `mlx_contiguous` at load (see `splitPackedGateUp`).

### SSM/GatedDeltaNet state init
`conv1dWithCache` sets `ssm.initialized = true` after conv update but BEFORE SSM recurrence state exists. Init code must check `ssm.ssm_state.ctx == null`, NOT `!ssm.initialized`. Used by both `mamba2Mixer` and `gatedDeltaNet`.

### Parameter-free RMS norm
mlx-c crashes on null/empty weight for `mlx_fast_rms_norm`. Pass `ones([dim], bfloat16)` for parameter-free norm. Affects GatedDeltaNet Q/K norm and Mamba2 group norm.

### Nemotron-H time_step_limit
Python defaults to `(0.0, inf)` (no dt clipping). `time_step_min`/`time_step_max` in config.json are NOT used by Python for SSM clipping. Only `time_step_limit` JSON array overrides.

### Speculative decoding (PLD + drafter) — overview

Two paths share a verify invariant: `cache.step = prompt_len + tokens_emitted`, t1 NOT in cache on entry, no pending state. Verify input is `[t1, draft[0..m-1]]` length `1+m`; full accept samples `new_t1` from `verify_logits[m]` (bonus prediction); partial accept rolls back via `KVCache.snapshot/restore` + `ssmSnapshot/Restore` and re-forwards `[t1, draft[0..accepted-1]]`. `accepted=0` still re-forwards `[t1]`. Pending correction sampled from *original* `verify_logits[accepted]` (NOT re-forward); index is `accepted` not `accepted-1` — off-by-one silently corrupts output, guarded by `tests/test_pld_equivalence.sh`.

### A speculative token cap belongs before state commit, not in the output loop (2026-08-12)

A review of the DFlash `max_tokens` fix exposed a class bug in every other block-returning decoder. Near a 17-token cap, PLD could return a five-token block after the Generator had already appended the whole block and advanced usage to 20. The scheduler then copied `gen.completion_tokens` while publishing the block's *first* token, immediately saw `>= max_tokens`, and stopped. The client received one token from that return while usage and internal state claimed five. DSpark, MTP, the Gemma drafter, and PLD all had that accounting shape; fixing only DFlash hid four copies of the same defect.

There are two required boundaries, in this order:

1. Every Generator path calls `capAcceptedForTokenBudget` after determining the natural accepted prefix but **before** choosing the pending correction and committing any representation of the round. That includes returned tokens, `generated_ids`, usage, trunk KV/SSM, DFlash captures, MTP history, drafter state, and DSpark's module-owned rings. DSpark is the important non-shell case: the cap is passed into `dsparkRoundWith` / the stochastic accept arm so `dsparkFinish` rolls the module back at the capped boundary. Slicing `round.tokens` afterwards is too late.
2. The scheduler publishes all five block modes through one `publishSpeculativeBlock` loop. It increments `slot.completion_tokens` once per pushed token and never imports the Generator's already-final whole-block count inside that loop. A source-scan class guard pins both sets: every speculative `next*` function must contain the pre-commit cap, and all five scheduler arms must call the one publisher.

The numeric DFlash equivalence test disables the economics gate with `dflash_min_accepted_per_round = 0`; otherwise it ran only three real DFlash rounds, fell back to serial, and spent most of the test proving serial equals serial. Gate behavior has its own policy tests and the live script now accepts either workload-dependent outcome. If the gate reports `runtime_disabled=true`, the emitted stats must prove `avg_per_round < gate_min`.

That review also found two DFlash lifecycle assumptions that were only true on the original M5/block-16/tool benchmark. The break-even threshold is now normalized by actual draft width: the non-thinking block-16 calibration is 2.0 accepted drafts/round and the thinking calibration is 1.0, each multiplied by `(effective_block_size - 1) / 15`. The request class comes from the server's fully resolved `enable_thinking` value on every streaming and non-streaming surface; `has_tools` is not a reasoning signal. Finally, a sticky runtime fallback grows the trunk with serial decode while the dormant assistant context stays at its last speculative boundary. `commitSlotIfApplicable` therefore stores a DFlash payload only when `dflash_ctx.absLen()` exactly equals `full_prompt.len + generated_ids.len`; a shorter context is omitted so the next request starts blind rather than restoring a mismatched assistant state.

**PLD** (`src/pld_index.zig`, `Generator.nextPld`): model-agnostic n-gram match in `prompt + generated`. CLI `--pld --pld-draft-len 5 --pld-key-len 3`; per-request `enable_pld`. `Generator.initWithOptions` clones prompt to `prompt_ids_owned` (caller-supplied freed before `nextPld`). Stochastic verify: draft as one-hot; `accept_prob = min(1, target_p[draft[i]])`, residual `max(target_p − one_hot, 0)` renormalized — preserves marginal per Leviathan. One-hot built via `pldOneHotRow` (no scatter).

**Drafter** (`src/drafter.zig`, `Generator.nextDrafter`): Gemma 4 only. 4-layer, hidden 256, no K/V projections — cross-attends into target's K/V via layer-type mapping (drafter sliding → target last sliding; drafter full → target last full). Loaded via `--drafter <dir>`. `block_size` auto-detected per target (E2B=2, E4B=4, 26B-A4B=4, 31B=8 — matches vLLM PR #41745) via `recommendedBlockSize`; override with `--draft-block-size`. Input: `concat([target.embed(prev) * sqrt(target.hidden), h_prev], -1)` → drafter hidden 256. Autoregressive within round (`block_size − 1` drafts), constant RoPE offset. Sparse `MaskedEmbedding` LM head (~2048 centroids, top-32 → ~4096 token logits of 262144). Linear weights pre-transposed at load.

**Validation**: `error.UnsupportedDrafterArch` (model_type mismatch), `error.DrafterTargetMismatch` (hidden_size or layer_types incompatible).

**`forwardCaptureHidden`**: `forwardStandard` and `forwardMoe` honor `capture_hidden`, slicing post-final-norm hidden at LAST position. Drafter seeds first `h_prev`; PLD uses during partial-accept rollback. Other forward paths (BERT, hybrid) leave it empty — drafter/PLD not wired there.

**Coverage**: PLD, drafter, and MTP dispatch on ALL FOUR HTTP surfaces — `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions` — in both streaming (`pickStreamMode`) and non-streaming (`nonStreamingViaScheduler` `use_pld`/`use_drafter`) modes. When adding an endpoint or dispatch path, wire BOTH flags through both modes and extend the engagement-count check in `tests/test_drafter_tools.sh` — two non-streaming call sites shipped with a hardcoded `use_drafter=false` for a month because output-equality tests can't see a silent fallback to regular decode.

**MTP (native multi-token prediction)** (`src/mtp.zig`, `Generator.nextMtp`): Qwen 3.5/3.6 checkpoints with a trained one-layer MTP sidecar (`mtp/weights.safetensors`, ~15 tensors, e.g. ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve — buildable from any Qwen 3.6 checkpoint via tests/build_mtp_sidecar.py) auto-load it and speculate with the model's own head. Forward: `fc(concat([rmsnorm(embed(tok)), rmsnorm(trunk_hidden_post_final_norm)]))` → one qwen3_5 gated full-attention layer with its OWN single-layer dense KV cache (partial RoPE at cache-relative offset) → `mtp.norm` → trunk lm_head. The head keeps a COMMITTED-HISTORY cache: entry j pairs (trunk hidden at position p, token at p+1), built chunk-wise during prefill (`ForwardCtx.capture_hidden_all`) and rebuilt each round — drafts append temporary entries from MTP-predicted hiddens; the commit STASHES the round's (tokens, TRUE verify hiddens) pair (`Generator.MtpHistStash`) and the NEXT round's first draft consumes it: one truncate to the stash origin + ONE merged multi-row head forward appends the committed history AND the first draft entry, byte-identical in RoPE offsets/append order to the old appendHistory-then-stepArr sequence (pinned by the merged-forward equivalence test in mtp.zig) but one head forward cheaper per round (~7 ms on the 27B: T(3) 76.3 → 69.5 same-protocol). Rounds with no successor (EOS/length/runtime disable) never pay for the append; a pending stash makes `mc.step` STALE, so the round origin is `mtpRoundOff0(stash, mc.step)`, never raw `mc.step`. `appendHistory` itself survives only for prefill history build. Verify invariant identical to PLD/drafter. Quant: MTP linears re-solve (bits, group_size) PER WEIGHT per call (`transformer.affineParamsFromGeometry` from packed-column geometry vs the activation inner dim — the 35B-A3B sidecar mixes 5/6-bit gs-128 q/k/v beside 4-bit gs-64 o and experts; load-time `inferBits` globals are only the degenerate-geometry fallback); fc + norms bf16. The MLP is a union: dense SwiGLU, or the 35B-A3B MoE layout (`language_model.mtp.` key prefix, mlx-lm `switch_mlp` split experts + shared expert + SEG) forwarded through the trunk's own `moeMLP` — live 73% per-draft on the 35B, temp-0 identical to AR. Sidecar discovery (`mtp.sidecar_rel_paths`) accepts `mtp/weights.safetensors` (native), root `mtp.safetensors`/`model-mtp.safetensors`, and `optiq/mtp.safetensors` (oMLX OptiQ). GOTCHA (delta-encoded norms — now auto-handled): the original Qwen repo (and OptiQ's export) store RMS-norm weights DELTA-encoded (the layer computes `1 + w`), while mlx-serve applies `rmsnorm(x) * w` with no +1 — so a head loaded without the fold is structurally broken (engages, ~0% acceptance, gate-falls-back to AR, passes equivalence+engagement). The server now AUTO-FOLDS at load: `mtpNormsAreDeltaEncoded` reads the negative fraction of the norm weights (delta ≈ 30-50% negative; folded RMSNorm scales are strictly positive), and `ownNorm` adds +1 when delta — a native (already-folded) sidecar has no negatives so it's left byte-identical (no double-fold). Worst case of a mis-detect is the runtime gate turning MTP off, never wrong output. SECOND OptiQ class (mixed-precision embed): `embedTargetTokens` must resolve the trunk embed table's quant params PER-WEIGHT via `transformer.computeQuantParams` (mirroring `Transformer.embedding`→`quantParamsHinted`), NOT the global `config.quant_bits` — OptiQ quantizes `embed_tokens` to 8-bit while the base is 4-bit, and dequantizing an 8-bit table as 4-bit crashes the whole server (`[dequantize] scales/biases shape mismatch`, a 36-row slice reads as `(36,1280)`); uniform-4-bit checkpoints (ddalcu, MTPLX) resolve to the same 4/gs64 so they're byte-unchanged. Both classes guarded by the delta-norm detect/fold + optiq-path tests in mtp.zig and `tests/test_mtp_equivalence.sh` (green on all three 27B variants: ddalcu/mtplx folded-untouched, optiq folded + 8-bit-embed, avg_per_round 1.0). Priority MTP > drafter > PLD; NOT subject to the prompt n-gram spec-gate (the trained head holds ~73% per-draft on fully novel content).

**The per-request default is ONE chokepoint: `server.defaultEnableMtp(mtp_loaded, is_moe, force)`** — `mtp_loaded and (!is_moe or force)`, called by all four surfaces (chat / completions / messages / responses). Never inline the policy at a new surface; an output-equality test cannot see a spec path that silently never engaged (the drafter-dispatch-hole lesson). MoE targets default **OFF** (the verify forward pays the expert-routing penalty, the same caution the drafter carries), which means **a MoE checkpoint that ships a head is unreachable from any client that doesn't send `enable_mtp:true` in the body** — Claude Code, llmprobe and curl all send nothing, so they silently measure plain decode (llmprobe reported `0.99× — none detected` on the 35B-A3B-MTP for exactly this reason). `--mtp` (`ServerConfig.default_force_mtp`, Settings ▸ "Also use MTP on mixture-of-experts models", off by default) flips that default for MoE; dense targets are unaffected. Measured on the 35B-A3B distilled (llmprobe, prefix cache off): engages ×33, speculative ratio **2.14×** on predictable content, but steady-state decode on NOVEL prose moves only 130.3 → 131.9 tok/s — i.e. the MoE default-off is right for chat and the flag is worth flipping for echo-heavy/agentic traffic. Measure per checkpoint (`./tests/llmprobe_smoke_test.sh`, the `mtp` cell); `--mtp` + `--no-mtp` together is incoherent and "off" wins.

**EV adaptive depth controller** (default ON; `MLX_SERVE_MTP_ADAPTIVE=0` reverts to the fixed windowed controller for same-boot A/Bs): each request tracks per-index CONDITIONAL acceptance EMAs `a[i] = P(draft i accepted | i−1 accepted)` (β 0.15, ~10-round legacy warmup) and plans every round via the pure `mtpEvPlanFor` — base `m_lo` = static EV argmax, extended to `m_hi` when the head's chain log-confidence on chunk A clears a cost-derived τ (ONE bounded sync at the chunk boundary). Invariants: (1) a single-chunk plan (`m_lo == m_hi`) is byte-identical in round shape to the fixed path — no confidence graph, no sync; (2) sticky-disable needs a FULL 16-round window of first-draft outcomes collected only while base depth is 1 (wider-base rounds reset it; demotion instant via EMA decay; `m_lo` climbs ≤ +1/round); (3) `MTP_EV_PRIOR` (0.85) sits ABOVE the ~77% average rate ON PURPOSE — deep indices are OBSERVED only when extension fires, so a realistic prior starves exploration forever (τ + the full-confidence horizon are the only gates, pinned by the exploration test). `--mtp-depth 0` = auto: cap 6 ordinarily, cap 8 ONLY for the calibrated G17-NAX fingerprint (see verifyQmm gotcha); explicit depths win (clamped to `MAX_DEPTH`), 0-sentinel plumbed through LoadParams so `lm.mtp_depth` logs stay truthful. `[spec-stats]` reports `drafted=`/`ext_rounds=` (per_draft_pct divides by DRAFTED tokens — depth varies per round). Guard: `tests/test_mtp_equivalence.sh` asserts ext_rounds>0 on a max-confidence echo AND `MLX_SERVE_MTP_ADAPTIVE=0` reverts to depth 3, zero extensions.

**Auto cap-8 fingerprint** covers the whole round, not just the trunk: also requires the native dense sidecar's affine-8/gs-32 (or 4/gs-32) q/k/v/o/MLP geometry + a materialized affine-3/gs-64 draft-only lm_head. `MLX_SERVE_MTP_DRAFT_HEAD_BITS=0`, a failed requant, or a compatible sidecar with different geometry keeps cap 6; explicit `--mtp-depth` wins.

**Round cost surface is qmm ROW-COUNT, not GDN** (`MLX_SERVE_MTP_TRACE=1` per-phase timings; `MLX_SERVE_GDN_UBENCH=1` attribution µbench): the GDN recurrence kernel is near-FLAT over multi-token widths while bare qmm×48 layers reconstructs the live forward ladder — the 27B prefills ~235 tok/s on every mlx engine because it's a dense 27B paying full qmm/token, NOT a GDN sequential-latency effect. DEAD ENDS (measured, don't re-try): chunked/WY GDN kernel ceiling ≤2.5% prefill; `mlx_compile`'d offset-free draft-chain dead-even live (MLX already batches launches); SAMPLED drafts lose to greedy same-session. EV cost constants live in `MTP_EV_DEFAULT_COSTS` (ordinary) / the calibrated `MtpCostProfile.g17_nax_*` (G17), selected per resident head, scalar-overridable via `MLX_SERVE_MTP_EV_COSTS="draft,lo,hi,sync"`. **Cross-request EV seeding** default-ON (`MLX_SERVE_MTP_EV_SEED=0` isolates): `Generator.deinit` publishes a healthy run's EMAs+base depth; the next request skips the warmup climb; `<8`-attempt/disabled runs never publish, so a creative sticky-disable can't poison the seed. **Rules**: never fit cost constants from a run whose realized `m_avg` you didn't check; thermal SOAK lies harder than drift (same-config warm readings can fall ~20% over 90 min of load — same-session ratios only). Prefill history FULL by default: `--mtp-history-window` opts into windowed capture but the A/B FAILED on the stock head (~14 acceptance pts at 64K — their adapter is TRAINED windowed, ours drafts from deep history).
The parity test feeds the reference implementation's trunk hiddens through our head and requires exact draft-token agreement, isolating head math from INT4 trunk-numerics differences (live drafts agree 25/26 with the reference; rejections are near-tie trunk argmax flips, not head bugs).

- **No snapshots across verify** (see the copy-on-write gotcha below): rollback anchors are SCALARS (`moe_seq_offset`, `cache.step`) + the verify pass's per-position SSM capture (`capture_ssm_seq`, same machinery as nextPld); partial accept = KV `truncate` (offset-only) + `ssmRollbackFromCapture` + the next h_prev SLICED from `verify_hidden_all` at index `accepted` — NO trunk re-forward. The MTP head's own cache likewise rolls back by `truncate(mtp_off0)`, never snapshot/restore. A pure-attention target (none exists today) keeps the snapshot+re-forward fallback; a GDN trunk whose capture didn't populate errors (`MtpRollbackUnavailable`) rather than silently corrupting.
- **One sync per stochastic round**: per-position probs come from ONE batched `probsAllPositions` (temp/top-k/top-p/softmax over `[1,1+m,V]` — a per-position loop pays 1+m separate ~vocab-sized sort kernels), accept probabilities are gathered with the LAZY draft-id arrays and read as one `[m]` vector, and a candidate correction for EVERY possible reject position is pre-sampled lazily in the same batch eval (only `corr[accepted]` is read; the rest are throwaway vocab-op work, far cheaper than a second synchronous softmax+categorical round-trip).
- **Cheap drafts**: drafts are GREEDY (argmax) by default — the one-hot acceptance rule is then exact rather than approximate, and acceptance loses its draft-side sampling noise (`MLX_SERVE_MTP_DRAFT_GREEDY=0` reverts); each draft's full-vocab projection goes through a DRAFT-ONLY 3-bit/gs64 lm_head requantized from the trunk head at bind time (`MtpModel.buildDraftHead` → `requantizeRows`, chunked so the dequant transient stays ~350 MB; `MLX_SERVE_MTP_DRAFT_HEAD_BITS`, 0 disables). Verification ALWAYS uses the trunk head, so the output distribution is untouched — at temp 0 the greedy acceptance rate dips (draft argmax disagrees more) but at sampled temps acceptance only needs a plausible draft, and the byte saving wins.
- Depth-controller thresholds assume the new cost model (demote 0.40 / promote 0.60): a rejected draft now costs ~2 ms of head work, not a 30-50 ms trunk re-forward. Sidecar files resolve through `mtp.sidecar_rel_paths` — native `mtp/weights.safetensors` first, then `mtp.safetensors`/`model-mtp.safetensors`

### A spec-verify forward is decode-shaped, not prefill-shaped — the mid-loop eval cadence must skip it (seq>1 ≠ prefill class)
The three prefill layer loops (`forwardStandardWith`/`forwardMoeWith`/`forwardHybridWith`) gate their periodic mid-graph `mlx_array_eval(h)` — which exists to bound GB-scale lazy transients during 2048+-token prefill chunks — on `is_prefill = seq_len > 1`. Spec-decode VERIFY forwards (PLD/drafter/MTP, seq 2–9) are multi-token, so they rode the same loops and paid the cadence: on the 64-layer qwen3.6-27B (`MOE_EVAL_EVERY_N_LAYERS = 4`) that was 16 synchronous pipeline drains per MTP round. Their transients are decode-scale (KB–MB) — seq-1 decode runs the whole 64-layer loop lazy with no evals at all, so a seq-7 verify bounding nothing new gains nothing and loses the pipeline. Fix: `Transformer.prefillEvalCadenceApplies(seq_len)` (`PREFILL_EVAL_MIN_SEQ = 32`) exempts verify-width forwards at all three call sites; the budget-driven per-layer flip (`prefillEvalCadence`) is untouched for real chunks. Note the honest epilogue: on the MTP round the removed drains mostly RELOCATED the wait to the round's read site rather than shrinking wall time (the GPU work itself dominates) — but the fix is still correct (verify graphs now stay fully lazy/fusable, short prompts < 32 tokens skip pointless evals) and it is what made the per-phase trace attribution truthful. Rule: any new multi-token forward that is not a real prefill chunk (verify, canvas re-encode, batched head forwards) must not inherit prefill-only cadence/eval behavior just because `seq_len > 1`. Guard: the `prefillEvalCadenceApplies` unit tests (transformer.zig).

### Speculative rounds must never hold a KV-cache snapshot across the verify forward (copy-on-write class)
`KVCache.snapshot()` refcount-shares the cache buffers. While ANY snapshot handle is alive, the next `update()`'s `slice_update` cannot donate the shared buffer and silently COPIES the whole thing — per layer, per round. At 64k ctx on Qwen3.6-27B that was ~4.3 GB of pure copy traffic per MTP round (16 full-attn layers × K+V), the dominant round cost and invisible at short contexts; the MTP head's own history cache paid the same tax per draft append (~268 MB each at 64k). Symptom signature: spec-decode round time GROWS with context far faster than the verify forward itself; AR decode (which never snapshots) outruns speculation at long ctx. Rule: rollback state for a spec round is SCALAR anchors + per-position captures + offset-only `truncate` — a snapshot is only acceptable on paths that will genuinely restore-and-re-forward (and those pay the copy tax knowingly). This is the runtime mirror of the "hot-cache restore must CLAMP" gotcha: both come from KV buffers being shared-by-refcount with in-flight writers.

**Auto-disable**: `logprobs > 0` and grammar-constrained sampling disable both. Tools disable NEITHER — agent traffic (tool results echoed into edits) is spec-decode's best workload; equivalence with tools is pinned by `tests/test_pld_tools.sh` and `tests/test_drafter_tools.sh` (~2.1× decode on file-edit tool calls, Gemma 4 E4B). PLD works on hybrid SSM (LFM2.5, Nemotron-H) — see snapshot null-state guard below; the drafter does not (verify forward hits the SSM-state issue). Drafter streams since spec-decode v3 (`pickStreamMode` routes streaming requests to `.drafter`). Drafter > PLD > regular priority when both enabled.

**Adaptive prompt-time gate** (`spec_gate_threshold = 0.01` in `server.zig`): n-gram repetition score on tokenized prompt (`pld_index.ngramRepeatScore`, 3-grams). If `score < threshold` AND user didn't set `enable_pld:true`/`enable_drafter:true`, the flag is silently disabled. Runs in all three request paths; chat-completions logs `spec-gate: ngram-score=X.XXX` once per request. v4 corpus validation: 9/9 correct decisions; threshold 0.01 cleanly separates "any 3-gram repeats" from pure-novel prompts.

**Runtime acceptance gate** (`RUNTIME_GATE_MIN_PER_DRAFT_RATE = 0.50`, warmup 5): when per-draft acceptance falls below 50% mid-decode, `Generator.spec_disabled_runtime` flips on (sticky). Subsequent calls short-circuit to `Generator.next`, which has a transition shim: when no pending logits/token, sync `forward([next_token_id])` to seed pending_logits. Pre-v26.5.6 the gate compared per-round against 0.30 → almost never fired; 0.50 cleanly cuts creative-content tail (22-47%) while leaving heavy-echo (84-97%) untouched. Does NOT save MoE+drafter regressions where per-draft is high but verify cost dominates — handled by MoE default-off in `serve()`.

**Default-on policy**: PLD is ON by default at the CLI (`main.zig` `enable_pld = true`; `--no-pld` disables). The drafter is opt-in via `--drafter <dir>`. While a drafter is loaded:
- Dense Gemma 4 (E2B/E4B/31B) drafter: `enable_drafter` defaults TRUE per-request; gates handle creative content
- MoE Gemma 4 (26B-A4B) drafter: `enable_drafter` defaults FALSE — verify forward MoE expert-routing penalty makes drafter regress at batch=1 even at 97.8% per-draft (every block_size tested). PLD remains default-on (1.43× echo). Per-request override still works.

### PLD/drafter long-greedy byte-divergence at INT4
AR (`next`) forwards `[1,1,d]` qmv; verify forwards `[1,K+1,d]` qmm. INT4 float reductions in slightly different orders → near-tie argmax can flip → divergence cascades. First ~30–80 generated tokens at temp=0 are byte-identical (equivalence tests live here); beyond that, paths may diverge char-by-char while both being mathematically valid greedy outputs. At temp ≥ 0.01 the Leviathan sampler preserves the target distribution → exact past 30 tokens. **For byte-stable long-greedy at temp=0 on INT4: `--no-pld`, no `--drafter`.** For chat/agent (temp>0) spec-decode is exact and free.

The same float-reduction issue compounds when **KV is also INT4** — see "KV cache quantization" below.

### KV cache quantization (`--kv-quant {off, 4, 8, turbo2, turbo4}`)
Group-wise affine quantization of K/V via `mlx_quantize`/`mlx_dequantize` (no new kernels). Storage swaps dense `[B,H,T,D]` bf16 buffers for a triple `(q, scales, biases)` where `q` is packed uint32 and `scales`/`biases` are per-group bf16; SDPA always reads dense data via `KVCache.denseView`, which dequantizes on the fly in quant mode. `--kv-quant` sets the **process default**; individual requests can override via the `kv_quant` body field on `/v1/chat/completions`, `/v1/messages`, `/v1/responses` (`"off"`, `4`, `8`, `"turbo2"`, `"turbo4"`). Memory: ~4× smaller at 4-bit (4.5 bits/elem including scale+bias overhead at group=64), ~2× at 8-bit. TurboQuant adds a Hadamard rotation before affine quant; `turbo2` halves bits-per-element again at the cost of an extra `[head_dim,head_dim]` matmul per K/V per token. Implemented in `src/kv_quant.zig` + `src/transformer.zig` (KVCache).

- **Equivalence thresholds** (`tests/test_kv_quant_equivalence.sh`, default 30/30; raise via env vars for stricter testing):
  - Gemma 4 E4B 4-bit weights: 30/30 passes; 8-bit KV stays identical past 60 in practice.
  - Qwen 3.5/3.6 MoE 4-bit weights (GatedDeltaNet + MoE): 4-bit KV passes 30 tokens. 8-bit KV diverges around token 41 from MoE+GDN float-reduction noise — same class as the INT4-weight long-greedy tail.
  - LFM2.5 8-bit weights (hybrid SSM): 4-bit KV diverges around token 12 — recommend `--kv-quant 8` for byte-stable long-greedy on this family.
  - Override per-arch via `KV_QUANT_FIRST_N_4BIT` / `KV_QUANT_FIRST_N_8BIT` env vars.
- **Compounding with INT4 weights**: Both the existing weight-quant divergence (PLD/drafter note above) and KV-quant divergence stack. For byte-stable long-greedy at temp=0 on INT4-weight models: prefer `--kv-quant 8` if you need a quant; `--kv-quant off` if you don't.
- **Drafter**: target's KV may be quantized; drafter cross-attends through `cache.denseView` so it never sees the quantized representation directly. Drafter's own cache stays dense. No special handling needed.
- **Snapshot / prefix cache**: snapshot/restore copy 6 array handles per entry instead of 2 (4 extra for scale/bias); hot prefix cache works unchanged because it operates on `KVCacheSnapshot` opaquely. Each `HotEntry` records its scheme; `findBestMatch` filters by `(prompt_ids, has_tools, scheme)` so per-request overrides never produce a cross-scheme hit.
- **TurboQuant (`turbo2`, `turbo4`)**: same affine-write/read path with a per-layer Hadamard rotation applied before quantization and undone after dequantization. `TurboState` builds `2 × num_layers` symmetric `[head_dim, head_dim]` bf16 matrices via Sylvester construction with per-layer column-sign flips (deterministic, no RNG seed). `head_dim` MUST be a power of two — caller passes via `KVCache.initWithConfigAndHeadDim`. State lives on `KVCache.quant_state` and refcount-shares through `snapshot`/`restore`. The rotation matters when inputs have outliers that would inflate per-group ranges in straight affine; on smooth data it can be slightly *worse* than straight affine because the rotation spreads tight local ranges into a wider global range.
- **1-bit TurboQuant**: not yet shipped. `mlx_quantize`/`mlx_dequantize` only support bits ∈ {2,4,8} natively, so 1-bit requires a custom pack/unpack. Land alongside the future fused-kernel work.
- **Extending the scheme** (e.g. fused quant-SDPA Metal kernel): the contract between cache and attention is `KVCache.denseView`. To add a new scheme:
  1. Add an enum variant to `kv_quant.Scheme`.
  2. (Optional) Add per-cache state (e.g. `quant_state: ?TurboState` for rotation matrices).
  3. Add `quantizeX` / `dequantizeX` functions in `src/kv_quant.zig`.
  4. Extend the `switch (config.scheme)` arms in `KVCache.update` and `KVCache.denseView`.
  SDPA call sites don't change. See top-of-file comment in `src/kv_quant.zig` for the worked TurboQuant example (now shipped).

### Hot prefix cache memory budget (`--prefix-cache-mem`)
Wave 1.B — the hot prefix cache used to cap on entry count alone; with 4 KB-ctx entries on Gemma 4 E4B that's an 8 GB worst case. `--prefix-cache-mem N{KB,MB,GB}` (default 2 GB) caps resident KV bytes; `commit` evicts LRU entries until `current_kv_bytes + new_bytes <= budget`. `0`/`off` disables the byte cap (count cap still applies). Each `HotEntry` records its bytes at commit time (sum of `mlx_array_size × mlx_array_itemsize` across keys/values plus the scales/biases triples in quant mode). Log line: `[hot-cache] resident=X.XX / Y.YY MB (E entries)` on every commit / eviction.

### head_dim-256 prefill: the msv_attn_p256 band kernel + the guards that stay load-bearing (long-context OOM class)
MLX's fused SDPA covers head_dim ≤ 128 in prefill (`sdpa_full`; `sdpa_vector` covers 256 for seq ≤ 8); **every Gemma-4 and Qwen3.5/3.6 checkpoint ships head_dim 256**, whose prefill otherwise rides the composed path that MATERIALIZES a `[heads, chunk, total_kv]` bf16 score tensor per layer (tens of GB/layer at long ctx — the uncatchable Metal OOM class). The self-contained flash-style kernel `msv_attn_p256` (transformer.zig, `mlx_fast_metal_kernel`; FA-2 online softmax, register-resident Q, float32 accum) covers hd-256 prefill via `fusedSdpa256Prefill` (null → composed fallback). Scoping is three regimes:
- **Sliding-band (Gemma local layers, `window > 0`): ALWAYS fused** (master kill `MLX_SERVE_FUSED_256=0`) — the band + block-skip run in-kernel so the GB-scale sliding mask is never built and out-of-band KV is never touched (composed has no answer). Output byte-identical; kernel-vs-composed one bf16 ULP.
- **Plain causal (`window == 0`): DEFAULT-ON via the kv-chunk dispatch budget** (`MLX_SERVE_FUSED_256_CAUSAL=0` restores composed) — see the IOGPU-preemption gotcha below for why every earlier ratio-gated variant lost live despite winning µbenches, and how the per-dispatch budget flips it.
- **seq ≤ 8 (decode + spec-decode VERIFY forwards): always declined** — sdpa_vector owns hd-256 there; shipping without this gate stole MTP verify forwards into a prefill tile walking the whole KV (decode collapsed at 4K). Rule: a prefill-shaped kernel must never accept decode-shaped work; mirror MLX's `supports_sdpa_full` seq gate.
K/V enter as cache VIEWS with strides (`ensure_row_contiguous=false` — a forced contiguous copy per layer erases the win). Test seam `transformer.fused256_override` forces both arms.
The three prefill guards key on ONE predicate, `transformer.prefillHeadDimFused` — guards and dispatch must never drift: with causal composed, score tensors still materialize for causal layers so `prefillEvalCadence` + `prefillMemoryNeeded` keep billing them; fused drops the score term (the quantized-KV `denseView` dequant term stays either way).
- `generate.boundedPrefillChunk` (score-formula budget, floor/grain 512) **deliberately caps on raw head_dim, ignoring the fused kernel**: the kernel removes the SCORE transient but a big chunk still scales the OTHER per-chunk transients (MoE gather, KV concat) — a big fused chunk buys a few % speed for tens of GB peak (a smaller Mac dies). Policy-split on `sliding_band_arch`: archs with NO sliding-band layers (qwen3_5/3_6) additionally cap the auto chunk at `min(base, 4096)` (composed causal block-skips, so small chunks are faster AND lighter); Gemma keeps formula-only (its band layers run fused and want few KV re-walks). `MLX_SERVE_PREFILL_CHUNK` is the uncapped escape hatch.
- `server.prefillMemoryNeeded` bills KV at the ACTIVE kv-quant width and bounds the MLP envelope by the SAME chunk via `generate.effectivePrefillChunk` (guard and prefill must never compute different chunks).
Symptom: Metal `Insufficient Memory` mid-prefill at a fraction of advertised context on an hd-256 model with the kill switch on (or a new unfused head_dim). Guards: `fusedSdpa256Prefill` parity/decline + `boundedPrefillChunk`/`prefillEvalCadence`/`prefillMemoryNeeded` unit tests.

### Verify-width split-K qmm kernel (spec-decode fast path; the MTPLX-turbo port)
Stock MLX qmm is tuned for M=1 decode (qmv) and large-M prefill (steel); the M=2..8-row shapes of speculative VERIFY forwards fall in a dead zone that underuses bandwidth. `transformer.verifyQmm` (kill switch `MLX_SERVE_VERIFY_QMM=0`) routes eligible 4-bit-affine qmms (gs {32,64,128}, bf16/fp16 x, K%64==0, N%4==0, 512 ≤ N) through two comptime-codegen'd Metal kernel families ported from MTPLX's `verify_kernels.py` (Apache-2.0, their measured design ledger in that file's header): **split-K** (threadgroup owns 4 output columns, K reduction split over 2 simdgroups for N≥4096 else 4, one barrier) for regular N at M 2..7 (M=7 uses a 2-column tile), and the **wide msg tile** (8 independent simdgroups/threadgroup, no barrier) for N ≥ 100000 — the lm_head class, where the tiny-tile grid thrashes the scheduler (measured 2.1× stock). Dispatched from the ONE `qmatmulBits` affine tail + `mtp.qLinearFwd`, so trunk verify, PLD/drafter verify on OTHER 4-bit archs (gemma!), small decode batches, and the merged MTP-head forwards all ride it. Hard-won rules: (1) **codegen NAMED SCALARS with LITERAL accumulator indices** — the array-indexed `Vec8 v[MROWS]` form of the SAME kernel stack-spills (measured 10× at M=6, exactly as the source ledger warned); (2) **M=8 is a spill/occupancy CLIFF** (T(7) round 636 ms vs 115 stock) — plain-SIMD eligibility caps at M=7 and `MTP_ADAPTIVE_DEFAULT_CAP` is 6 so verify stays seq ≤ 7 (the NAX lane below is the sanctioned way past it, on hardware that has the units); (3) **µbench wins can still lose in-context** and vice versa (the msg tile wins iso at lm_head but loses to split-K in-context at regular N — mirror of the msv_attn_p256 lesson) — same-boot live traces decide; (4) a stale `zig-out/bin/mlx-serve` after `zig build test` reproduces the exact previous run's numbers — rebuild the EXE before every live A/B. Numerics: fp32 accumulate in a different order → bf16 tail-ULP class only; guarded by the `verifyQmm` parity test (M 2..7 × three shape classes + ragged-N msg tail + probe-forced M=1/8/16/17 fall-through + self-gating NAX rows M {8,9,12,16}), the live MTP equivalence suite, and gemma-4bit PLD byte-equivalence (the kernel engages on its verify too).

**NAX m16 lane (M5-class)** — `transformer.runVerifyQmmNax`: MTPLX's `nax_verify.py` m16 tensor-ops tile (MetalPerformancePrimitives `matmul2d` on the per-core matrix units; provenance DFlash → dflash-mlx → MTPLX, Apache-2.0, kept in the source comment). Routes M 8..16 (zero-pad to 16 rows, slice back) when `verifyQmmNaxAvailable()` (arch prefix `applegpu_g17` + macOS ≥ 26.2) and K%256==0/N%32==0; M 2..7 keep split-K/msg even on M5 (`MLX_SERVE_VERIFY_QMM_NAX_MIN_M` A/Bs routing 5..7 to NAX). Lane selection is the pure `vqmmLaneFor` (hermetic table test). Rules: (a) **the kernel object is NEVER BUILT — not just never dispatched — where the probe is false** (pipeline creation can fail off-G17; the `vqmm_nax_probe_override` seam is forced FALSE only on non-M5); (b) auto depth 8 needs the FULL homogeneous-affine calibration (token embed + every trunk projection + lm_head) AND NAX live for both M=8 and M=9 — else cap 6; explicit depths win. Parity/µbench rows SELF-GATE on the probe (M {8,9,12,16} × gs {32,64,128} vs fp32 dequant truth); `zig build test -Dtest-filter=verifyQmmNaxAvailable` prints `[nax-probe] arch=… available=…`.

**Eligibility predicates adopt every matching shape:** the verifyQmm kernel (tuned on qwen/MTP) rides the gemma-4-E4B drafter verify at M=5 and measured a small NET LOSS there (kill-switch A/B at identical engagement — kernel cost, not acceptance). **Rule: A/B each adopted shape on its own model, not just the one you tuned for.** (The bench-comparison rules that first mis-flagged this as a false regression — same-methodology CSVs only, spec cells need cross-run/boot-order samples, kill-switch A/Bs beat cross-version diffs — live in the `/bench` skill + CLAUDE.md ## Releases.)

**A GPU-kernel parity test asserts NO-WORSE-THAN-REFERENCE against fp32 dequant ground truth — never AGREEMENT with another kernel (green-here/red-on-CI class).** Both paths accumulate fp32 in a DIFFERENT ORDER and round to bf16, so two CORRECT kernels can differ from EACH OTHER by more than a tight tolerance — a kernel-vs-kernel bound (`max_rel<=0.02`) passed 10/10 on the dev Mac and RED on the first CI GPU (different rounding) on a provably-correct kernel. Fix: `transformer.expectVerifyQmmNoWorseThanStock` measures BOTH against the same 4-bit weights dequantized to fp32 (quant error cancels; only the arithmetic is under test) and requires the new kernel ≤ stock's error — machine-independent, and STRICTLY STRONGER (catches a 1% scale error, a dropped partial sum, a single zeroed output). Rule: never bound a new kernel against another kernel's rounding; bound both against ground truth.

### Hot prefix cache on hybrid SSM models retains far more than it reports
Measured live (2026-06-19, Qwen3.5-4B = 8 attention + 24 GatedDeltaNet layers, 16 GB Mac): with the cache ON, MLX `active_memory` climbed ~2.2 GB **per agent turn** and never released (model 2.2 GB → 6.8 GB after 3 turns); with `--prefix-cache-entries 0` it stayed flat at the model size and only `peak` rose (the transient prefill spike, released after). Two facts: (1) each commit `captureSsmCheckpoint`-snapshots the per-position conv/SSM state of all 24 linear-attention layers AND refcount-SHARES the arrays, so the real retained allocation (~4.5 GB at 3 entries) is ~3.4× the reported `[hot-cache] resident` (1.3 GB) — the byte cap is checked against the under-count, so it never evicts enough. (2) `--kv-quant` does NOT help: it compresses only the 8 attention layers' K/V, not the dominant SSM/conv state. Reliable lever is the ENTRY count, not the byte cap. **Swift-launcher gotcha**: the server's `prefix_cache_capacity` default is **32** (raised from 1 to stop llama eviction thrash), but `ServerOptions.toCLIArgs` historically only emitted `--prefix-cache-entries` when `!= 1` (assuming server default 1) — so the app silently launched 32 entries and filled 16 GB Macs. Fixed: the flag is ALWAYS emitted, RAM-clamped via `ServerOptions.ramCappedPrefixCacheEntries` (≤18 GB → 1, ≤36 GB → 8, else uncapped) and surfaced in SettingsView. The under-counting accounting bug itself is still open — fix `snapshotBytes`/`ssmCheckpointBytes` to reflect true retained allocation (materialize snapshots with `mlx_contiguous`, or measure parent-buffer footprint) so the byte cap actually bounds memory.

### PLD on hybrid SSM (snapshot null-state guard)
`SSMCacheEntry` has two slots (`conv_state`, `ssm_state`) populated by different layer types: LFM2's `gated_conv` writes only `conv_state` (sets `initialized=true`) and never touches `ssm_state`. `mlx_array_set` with null source aborts via mlx-c's default handler (`exit(-1)`). `ssmSnapshot`/`ssmRestore` and `PrefillCache` save/restore must check each field's `.ctx != null` independently — `initialized` alone insufficient. This was the previous "off on hybrid SSM" auto-disable; lifted once per-field guard landed.

### mlx-c iterator refs: `iterator_next` hands you a +1 you must transfer or free (phantom-ref leak class)
`mlx_map_string_to_array_iterator_next(&key, &value, iter)` gives the CALLER a reference in `value`. Either transfer that exact handle into your container (`map.put(key, value)` — the model.zig `loadSafetensorsFile` pattern) or free it; **copying it via `mlx_array_set(&owned, value)` and dropping `value` on the floor leaks a phantom +1 on every tensor** — the container's later `mlx_array_free` decrements to 1, never 0, so the buffers are immortal. Live bite (2026-07-03): `ltx.loadComponent` did exactly this, so EVERY video load→generate→unload cycle leaked the whole materialized engine (~18 GB per Generate click with keep-off; RSS hit 100 GB) — and it presented as an unload/cache bug, not a load bug. Diagnosis pattern: `/props` `active_bytes` NOT dropping after `unload-model` means leaked *handles* (allocator-cache growth drops on free + `mlx_clear_cache`; phantom refs don't). Guarded by the hermetic `loadComponent releases every tensor on deinit` test (ltx_video.zig, verified red-on-revert). Rule: any new `iterator_next` loop must account for the reference it was handed, and free the fresh empty `value` handle on the break path.

### A decode loop whose buffer sizes never repeat must release the MLX allocator cache (cache-growth leak class)
The complement of the phantom-ref class above, and its diagnostic mirror: **`active_bytes` FLAT while `phys_footprint` climbs = allocator-cache growth, not leaked handles.** `mlx_array_free` returns a buffer to MLX's allocator cache, not to the OS; the cache only serves a later allocation of the SAME size, and its limit on a 128 GB Mac is ~115 GB. So any loop that frees large buffers whose sizes never repeat grows the process footprint without bound while every handle is accounted for. This is why every AR decode path in `generate.zig` carries `if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();` — that line is load-bearing, not hygiene.
- Live 2026-07-09 (soak): `diffusiongemma-26B-A4B` (14 GB model) reached a **92 GB** phys_footprint. `diffusion.zig`'s canvas loop was the ONE decode path with no periodic clear, and it is the hungriest: each of up to 48 denoising steps frees a `[1, canvas_len, 262144]` f32 logits array (67–268 MB), its scaled copy, soft embeddings, and attention scores over a KV that grows by one canvas per commit — then the next REQUEST restarts at a different prompt length, so a freed buffer is almost never the size the next allocation wants. Reproduced exactly: ten requests with *growing* prompts drove phys 27.6 → 35.8 GB with `active_bytes` pinned at 22.32 GB; ten requests with an *identical* prompt stayed flat (same sizes → cache hits), which is why single-shot testing misses it entirely.
- **STEADY STATE and PEAK are two separate bugs, and fixing the first hides the second.** Releasing once per committed canvas stops the across-requests growth, but the *chunked encoder* still piles a whole prompt's transients into the cache inside ONE request: each chunk's buffers are sized by the KV offset, which grows every chunk, so the cache accumulates the SUM over chunks — quadratic in prompt length. With the per-canvas release already in, a 12,022-token prompt still grew phys 24.5 → 34.4 GB during its 20 s prefill (`active_bytes` moved only 22.9 → 24.6 GB, so ~8.4 GB was cache), and a 36,560-token agent turn peaked at **96.5 GB** — then *recovered*, which is exactly why it reads as "scary spike" rather than "leak".
- Fix, at BOTH cadences (mirroring `generate.zig`, which already clears once per prefill chunk right after the per-chunk eval AND every 256 decode tokens): `Runner.encodeTokens` calls `releaseCache()` at the end of every chunk, and `Runner.nextCanvas` declares `defer self.releaseCache();` FIRST so it runs LAST, after every transient's own defer has freed it into the cache. Measured on 36,022 tokens / 18 chunks: peak **96.5 → 42.3 GB** (the residue is real KV growth — `active_bytes` climbs 24 → 32 GB — plus one chunk's transients); decode cost is noise (≈30.8 vs 31.7 tok/s).
- **Rule: the repro must VARY the shape that drives allocation size** (prompt length, KV length, batch) — a benchmark that replays one fixed prompt proves nothing about cache growth. **Rule: sample the footprint DURING the request, not after it** — a post-request reading is taken right after the release and shows a flat, healthy number while the intra-request peak is 3× higher; use `vmmap -summary <pid>` (`Physical footprint (peak)` is the high-water mark since launch) against `/props` `active_bytes`. **Rule: any new decode/denoise/prefill loop owes a periodic `mlx_clear_cache()`, once per CHUNK and once per emitted block** — the symptom is a footprint that scales with prompt length, is invisible to `ps` RSS (Metal buffers) and to `active_bytes`, and (if only the outer cadence is fixed) returns to normal after each request.
- Guarded by two `DIFFUSION_TEST_MODEL` tests in diffusion.zig: `each committed canvas releases the MLX allocator cache` (red-on-revert: `1633.9 MB` cached after canvas 1 vs a 256 MB bound) and `prefill releases the allocator cache once per CHUNK, not once per prompt` (red-on-revert: `6151 tokens, 4 chunks, 1 cache releases`). The second pins the CADENCE via `Runner.cache_releases`, not the post-prefill residue — a trailing clear leaves the cache empty at the end either way, so a residue assertion cannot see the peak regress.
- NOTE: the per-request `Runner` (which materializes the ~1.5 GB dense embedding table) is NOT leaked — slots are created per request in `Scheduler.submit` and destroyed per request via `complete` → `cleanup_queue` → `Slot.deinit`, which frees it. `Slot.deinit` running "at slot teardown" IS per-request.

### O(specials × text) special-token scans (quadratic-tokenize class)
Splitting text around special tokens by re-searching the WHOLE remaining text for EVERY special token is O(specials × text) — invisible on Gemma 4 (24 specials) and lethal on gemma-3 (6,415): ~12 s of pure CPU per 66 KB prompt, re-paid on every tokenize-cache miss (so every multi-turn extension), reading as "slow model" when it's the scan. This class shipped TWICE — `Tokenizer.encode` (tokenizer.zig) and its duplicate `encodeWithSpecialTokens` (chat.zig) — and the live symptom only surfaced on the copy the chat path actually used, after the first was already fixed. Both now ride ONE fast path: `Tokenizer.encode`'s single left-to-right pass over first-byte-bucketed candidates (bucket sorted by descending length, so the first `startsWith` hit is the longest match — semantics identical to the old scan: earliest occurrence, longest at a position); chat.zig's wrapper just delegates. Rules: (1) never add a new special-token splitter — call `Tokenizer.encode`; (2) any per-position loop over a token/vocab COLLECTION needs a first-byte (or equivalent) index — vocab-derived collections can be thousands of entries; (3) diagnosing "slow first token": the response's `timings.tokenize_ms` isolates render+tokenize from prefill, and `/tokenize` isolates the tokenizer from the Jinja render. Pinned by the `encode special-token scan` characterization test (tokenizer.zig).

### mlx `Copy`/`contiguous` are view ops — a "copy" that must not alias needs an arithmetic kernel
`mlx_copy` is NOT a deep copy (MLX's Copy primitive shares the input buffer), and `mlx_contiguous` no-ops whenever the view's strides read as row-contiguous — which a size-1-leading-dim slice does. So neither breaks the alias to a slice's parent buffer, and "materialize this slice so the parent can be freed" silently retains the parent. This was the hot-cache under-count: SSM checkpoints refcount-shared conv/ssm states that were slices of whole prefill-chunk buffers (~3.4× more retained than reported). The working materializer is `transformer.materializedOwnedCopy` — add a same-dtype scalar zero (a real kernel with a freshly allocated output; donation can't alias because the caller still holds the input) — followed by an eval so the lazy copy node itself doesn't pin the parent. Rule: any slice that OUTLIVES its parent's intended lifetime (cache entries, checkpoints, anything committed to a store) goes through `materializedOwnedCopy` + eval; pointer-range test pattern in `captureSsmCheckpoint materializes state copies` (transformer.zig). Related trap in the same family: `mlx_save_safetensors` silently APPENDS ".safetensors" when the path lacks it — name files with the real extension or every later stat misses.

### mlx-c API changes
mlx-c 0.6.0 added a `global_scale` param (may be null) to `mlx_dequantize` between `mode` and `dtype`. FFI in `mlx.zig` must match installed header. When upgrading, diff `lib/mlxc-src/mlx/c/ops.h` (the pinned submodule) against `extern "c"` decls.

### The Homebrew mlx bottle ships with NAX silently disabled — we self-build from pinned submodules (silent-fallback class, 2026-07-19)
MLX's M5 neural-accelerator (NAX) kernels are gated at CMake configure time: Metal 4 support AND macOS SDK ≥ 26.2 AND `CMAKE_OSX_DEPLOYMENT_TARGET` ≥ 26.2. When the gate fails, the kernels are skipped AND `MLX_METAL_NO_NAX` is compiled into the dylib, making `is_nax_available()` return **false unconditionally** — stock ops (quantized_matmul, gather_qmm, steel gemm, SDPA) never dispatch NAX variants, even on M5 hardware. The gate fails **silently**: upstream only added a configure-time *warning* after v0.32.0 (commit 4367c73). The brew bottle fails it because `Formula/m/mlx.rb` pins `MACOSX_DEPLOYMENT_TARGET` to the **build host's point release** and Homebrew's Tahoe builders run 26.0 — verified by `strings mlx.metallib | grep -c nax` → 0 and the AIR target `macosx26.0.0` in the shipping bottle (re-poured 2026-07-08). Flip side: the day brew's builders hit 26.2+, a rebottle would flip stock-op NAX ON with no formula change and no announcement — shifting the perf baseline under the M5 EV calibration (`MtpCostProfile.g17_nax_*`, verifyQmm lane margins, bench CSVs).
- **Fix**: mlx (`lib/mlx-src` @ v0.32.0) + mlx-c (`lib/mlxc-src` @ fba4470) are pinned submodules built by `scripts/build-mlx.sh` with `-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2` into `lib/mlx/{lib,include}` (gitignored stage, `.version` stamp = SHAs + target, idempotent). build.zig links the stage (`addMlxLib`, before `/opt/homebrew/lib` so a leftover brew mlx-c can never win the link; `use_pkg_config = .no`), `verifyMlxStage` errors with the fix when unstaged, `verifyBrewDeps` keeps only webp. Release/app bundling copies from `lib/mlx/lib` (all `*.dylib` — libjaccl is an @rpath sibling — plus the metallib); install names are now `@rpath/...` so the existing otool-discovery rewiring keeps working.
- **The assert IS the point**: every failure mode (wrong SDK, deployment-target regression, missing Metal Toolchain — Xcode 26 ships the Metal compiler as a separate `xcodebuild -downloadComponent MetalToolchain`) surfaces as a quietly NAX-less metallib. `build-mlx.sh` hard-fails when the built metallib lacks `*_nax` or minos < 26.2; `tests/test_mlx_staged_nax.sh` re-checks the stage + that the built binary links no brew mlx.
- **mlx-c pin must be a brew-proven pair**: the v0.6.0 *tag* does not compile against mlx 0.32.0 (fft API: `FFTNorm` arg + `Shape` vs `std::vector<int>`); brew's 0.6.0_3 = v0.6.0 + exactly the 4 commits on mlx-c main (#110 #111 #112 #114), so the submodule pins `fba4470` — byte-identical source to what `src/mlx.zig` was validated against. When bumping mlx, bump mlx-c to whatever brew (or upstream CI) proves compatible, and re-diff the headers.
- **Consequences**: deployment target 26.2 is contagious — bundled dylibs refuse to load below macOS 26.2, so `LSMinimumSystemVersion`, `Package.swift` platforms, and the README floor all moved to 26.2 together. Compiling needs NO M5 (kernels compile to AIR; the runtime gen ≥ 17 probe gates dispatch), so M1–M4 behavior is unchanged and CI's virtualized M1/M2 runners can build it; only *measuring* NAX effects needs real M5 hardware (community M5 channel). Building from source also means new same-methodology bench baselines — pre-submodule CSVs were measured on the brew runtime.

### A non-absolute path to `openDirAbsolute` is `unreachable` → ReleaseFast UB that miscompiles the CALLER
`std.Io.Dir.openDirAbsolute` asserts `path.isAbsolute(path)` — and `assert` is `unreachable` on failure. In Debug that's a clean panic; in **ReleaseFast `unreachable` is undefined behavior**, and the optimizer is free to assume the assertion held. Live bite (feature/Any2Any headless boot): `isGgufPath(io, model_dir)` was called with `model_dir == ""` (headless `--serve` with no `--model`); `openDirAbsolute("")` hit the assert, and the resulting UB made the optimizer **eliminate a LATER branch in `main()`** — the `if (model_dir.len == 0)` headless dispatch was silently dropped, so the server fell through to `parseConfig("")` → `FileNotFound`. Symptom signature: a branch you can SEE in the source is provably-not-taken in ReleaseFast (its string literals are absent from `strings <binary>`), while a Debug build panics inside a stdlib `assert`. The bug is NOT where the crash/miss appears — it's the earlier UB poisoning the whole function. Rule: guard every `openDirAbsolute`/`openFileAbsolute` call against empty / non-absolute input (`if (path.len == 0 or !std.fs.path.isAbsolute(path)) return ...`) BEFORE the call; never feed it a possibly-empty path. When ReleaseFast and Debug disagree on control flow, suspect `unreachable`-class UB (failed assert, `@intCast` overflow, null `.?`) upstream — bisect by building Debug, which turns the UB into a located panic.

### Batched decode fed the RAW quantized KV buffers to SDPA — first concurrent request under `--kv-quant` killed the server (denseView-contract violation, 2026-07-20)
`forwardBatchedDecode` (the ≥2-slot decode kernel) read `slot_ctx.cache.entries[layer].key_view`/`value_view` directly when gathering per-slot KV for `padAndStackBatchedKV`. In dense mode those alias the dense views, so everything worked; under `--kv-quant` they hold the PACKED quantized words (hd 256 at 8-bit → last dim 64 uint32s, scales/biases in separate buffers), so the first tick where two requests decoded together died with `MLX error: [scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape (1,8,1,256) for keys shape (1,1,22,64)` — an uncatchable process kill, mid-llmprobe run. Every single-request path was green because `active.len == 1` routes through the legacy tick, which already honored the contract; `llmprobe_smoke_test.sh`'s KV-quant crash cell also only sends serial requests, so it couldn't see it. Presented as "server crashes immediately with kv cache on" — it actually crashed at the first CONCURRENT decode, which llmprobe reaches within seconds.
- **Fix**: the batched layer loop now takes one `KVCache.denseView(view_layer)` per slot (dequantizes for quant schemes, free alias in dense mode, per-slot scheme so mixed per-request `kv_quant` batches are fine), uses it for both the kv_max gather and `padAndStackBatchedKV`, and deinits the views after the layer's SDPA is enqueued.
- **Repro without a race**: `MLX_SERVE_FORCE_BATCHED=1` + `--kv-quant 8` + ONE request routes N=1 through the batched kernel deterministically. That's the regression guard — the "batched-kernel x kv-quant crash guard" section in `tests/test_batched_equivalence.sh` (server must stay healthy, return a completion, and log no `MLX error`). Verified red pre-fix with the exact production signature.
- **Rule**: the kv-quant contract ("attention always reads `KVCache.denseView`") applies to EVERY forward path, and a NEW forward path (batched kernel, future fused variants) must be exercised at least once under a quant scheme before it ships. Raw `entries[].key_view`/`value_view` reads outside kv-cache internals are a red flag in review.

### The fused-causal live loss was macOS IOGPU preemption of long dispatches — a kv-chunk dispatch budget flips it to a win (2026-07-22)
A single `msv_attn_p256` CAUSAL dispatch scanning the whole KV monopolizes the GPU long enough at high kv_len to trip **macOS IOGPU interactivity preemption**, the penalty scaling with dispatch wallclock — invisible to µbenches (isolated kernel) and short-kv ratio gates, which is why the arm won every µbench yet lost every live ratio-gated A/B monotonically. Fix (ported from oMLX `qwen35_fa256_attention.py`): cap per-dispatch work at `batch·Hq·qL·keys ≤ budget` (`MLX_SERVE_FUSED_256_BUDGET`), splitting the key axis into BK-aligned dispatches with the FA-2 online-softmax state (m/l/unnormalized O) carried through fp32 buffers — register-precision-exact, so BIT-IDENTICAL to single-dispatch (pinned by a test + a dispatch-count engagement seam). Implementation trap: MLX's `metal_kernel` codegen binds sub-4KB inputs in the `constant` address space and larger in `device`, so carry reads must use direct indexing (`o_in[base + i]`), never a typed `const device float*` — a compile error only on the arm you didn't test first.

### Stock qmm_t at prefill widths is a ~10% dead zone — dequant + steel GEMM route (2026-07-22)
Stock MLX qmm_t picks a 32×32×32 tile that stalls at M≥2048 on qwen shapes. `prefillDqGemm` (M≥2048, affine 2/3/4/5/6/8, `MLX_SERVE_PREFILL_DQ_GEMM=0` kill) dequantizes weights to a bf16 [N,K] transient and runs the dense steel GEMM near-peak — the per-call dequant costs <1 ms and repeats sizes per layer (allocator-cache-friendly); decode/verify widths never route; numerics differ only by bf16 weight rounding (no-worse-than-stock pinned). Trap: `boundedPrefillChunk`'s score-budget formula once shrank the 64K chunk below the route's 2048 floor — it silently never engaged at 64K (the ladder's weakest rung) until the fused-causal chunk policy (`min(base, 4096)`, no formula shrink) replaced it.

### Sharpened-stochastic MTP drafting (Lightning scheme) LOSES to greedy argmax on low-entropy content — negative result, machinery kept opt-in (2026-07-22)
Ported oMLX Lightning's scheme faithfully (drafts from a fixed sharper dist temp 0.6/top-p 0.95/top-k 20, full Leviathan `min(1,p/q)` accept, `normalize(max(p−q,0))` rejection — distribution-exact for ANY q, toy-vocab test). On low-entropy code/agent content greedy WINS: sharp acceptance is `1 − TV(p,q)`, greedy is `p(argmax_q)`; when the target is near-deterministic (temp-0.6 code sharpens p toward one-hot) `p(argmax) ≈ 1` while TV is dominated by the head's spread mass. Their "collapses to 10–20%" claim is about HIGH-entropy prose (the opposite regime). Keep it wired (`MLX_SERVE_MTP_DRAFT_GREEDY=0`) — the exact-residual machinery is correct and may win on prose.

**MTP round pipelining — the CPU graph-build was the recoverable overhead; our emit gap is ~0.03 ms so pre-draft buys little beyond early dispatch.** Three landed levers, each kill-switched and BIT-IDENTICAL to its off state (lazy sampling ops bind their PRNG key at graph BUILD): (1) **early dispatch** (`MLX_SERVE_MTP_EARLY_DISPATCH=0`) — `mlx_async_eval` the draft chain as soon as Phase 1 builds it, so it runs while the CPU builds Phases 2–4. (2) **cross-round pre-draft** (`MLX_SERVE_MTP_PREDRAFT=0`) — `nextMtp` tail builds+dispatches the next round's chunk A (plan after the EV update == head-of-round); mirrors oMLX `_step_mtp` but our scheduler has no Python-sized emit window, so it ≈ early-dispatch on totals (kept for the cheaper EV boundary sync). A `MtpPreDraft` owns every handle; consume asserts stash-XOR-predraft. (3) **GDN capture-tail trim** — the seq kernel emits `state_out` from registers and never writes `state_seq[T-1]` (partial accept reads ≤ T-2), so the final state is no longer a slice VIEW pinning the whole [T,…] buffer. Residual vs oMLX is GPU work (their round ≈ their AR forward), not scheduling.

**EV controller under honest costs — two structural traps fire once marginals stop being cheap.** (Refit cost constants only on a SATURATED sweep whose realized `m_avg`==depth — ladder prompts demote-flap and poison the fit; echo pins it.) Trap 1 — **two-chunk plans pay a mid-pipeline boundary sync the cost surface can't see** (the chunk-A sync blocks on the still-running head chain + confidence graphs): fix = the extension **dry-spell gate** (`mtpExtDryAllows`: ~16 dry rounds → short single-chunk cooldown → fresh trial; `MLX_SERVE_MTP_EXT_DRY=0`), fed by REALIZED extension rate never priors, cooldown SHORT vs a request's round count (64 swallowed a 160-token request's echo stretch). Trap 2 — **the horizon check deadlocks on an unobservable EMA**: `a[m_lo]` updates ONLY when extension fires, so a value dragged cold under an earlier workload closes the horizon FOREVER (pure echo runs ext_rounds=0). Fix: when the base pays (`best_r > MTP_EV_EXPLORE_MIN_R = 1.10`) one extension position stays reachable at the clamped tau. Pinned by the equivalence echo test + `mtpEvPlanFor` unit tests.

### A slice handle wrapped in `contiguous` and never freed pins its PARENT's buffer (MageFlow 2.2 GB-per-megapixel leak, 2026-07-25)
The third member of the family above, and the one that hides best: not a phantom +1 (the handle is legitimately ours) and not allocator-cache growth (`active_bytes` climbs with it). `mage_flow.sliceSeq` was
```zig
var o = mlx.mlx_array_new();
try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
return contig(o, s);   // `o` is never freed
```
— correct output, every parity fixture green, and six helpers across the DiT, text-encoder, ViT and VAE paths written the same way. **An mlx slice is a VIEW: keeping its handle alive keeps the whole parent buffer alive**, so `jointAttn`'s two calls (txt slice + img slice off the same `flat`) retained the img+txt pair the block was handed — ~47 MB at 1024², × 12 blocks × 4 steps = **~2.2 GB per megapixel per generation**, linear in output pixels within 6% across five sizes from 0.26 to 1.57 MP. It compounded across generations and SURVIVED `/v1/unload-model` (`mlx_active` 3.60 → 7.19 → 10.79 GB over three load→gen→unload cycles); a normal session reached 40+ GB.
- **What made it slow to find**, and the ladder that worked: RSS is BLIND to Metal buffers — it sat at 8.48 GB the whole time while 10 GB leaked, so `ps` says "healthy" and only `/props` `active_bytes` + `vmmap -summary` see it. FLUX klein is the clean CONTROL (returns to exactly 0.00 GB per cycle) — running the identical harness against a sibling backend is what proved the bug was ours and not MLX's. Stage instrumentation localized the gain to `Dit.forward`, then per-block marks to a flat ~47 MB per `ditBlock`. The decisive step was forcing `mlx_array_eval` on `img`/`txt` after every block: memory still grew, which RULES OUT lazy-graph retention and says the bytes are held by a live reference — the whole remaining search is then "which handle is never freed", which greps out in minutes. Quantization was a red herring (the dense bf16 checkpoint leaked byte-for-byte the same, 3.60/7.19/10.79 — identical figures mean activation-shaped data, not weights).
- **Fix**: one `sliceContig(x, lo, hi, st, s)` owns the intermediate (`defer mlx_array_free(o)` before `return contig(o, s)`), and all six helpers delegate to it — the pattern now exists in exactly ONE place. Numerics are untouched: the same seed produces a **byte-identical PNG** before and after. `krea.zig`/`flux.zig` were already correct (`defer free(out); return contig(out, s)`), which is why only MageFlow leaked.
- **Rule**: a helper that materializes a view owns the view — free the intermediate, don't just wrap it. `mlx_clear_cache` is NOT the fix for this class (that's the cache-growth one above); if `active_bytes` itself climbs, you are holding handles. Prefer one shared slice-and-materialize helper per file over N hand-rolled copies: this shipped six times in one file because each site was written independently.
- Guards: `tests/test_media_gen_memory.sh` (varies the size-driving shape across generations — a fixed-size replay cannot separate a leak from size-keyed caching — and asserts three load/gen/unload cycles return to the pre-load baseline; red-on-revert at +3.18 GB across four generations) and the hermetic `materializing helpers hand back every array they take` in mage_flow.zig, which calls each helper with a source built and freed INSIDE the loop and asserts `mlx_get_active_memory` returns to baseline. **The input must be rebuilt per iteration**: a caller-owned source that outlives the call keeps the parent alive anyway, and the first version of that test passed against the broken code for exactly that reason.

## GDN blocked-prefill kernel: hardcoded bf16 vs an f16 checkpoint (2026-07-25)

`./mlx-serve --serve --model=~/.mlx-serve/models/…/Nanbeige…` died mid-request with

```
MLX error: [metal::Device] Unable to build metal library from source
utils.h:476:19: error: cannot initialize a variable of type 'const device bfloat *'
                       with an rvalue of type 'const device float *'
const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
```

Two independent bugs stacked, and the first hid the second.

**It was not the model in the command.** `main.zig`'s flag loop takes values as a
separate argv token, so `--model=<path>` matched nothing and was silently
dropped (see docs/gotchas/server-http.md). The server went headless, and on the
first request auto-picked `prism-ml/Ternary-Bonsai-27B-mlx-2bit` as the default
chat model. That is what crashed. Nanbeige never loaded at all — its
`model_type` is `nanbeige`, which discovery already skips.

**The crash: the checkpoint is F16, not bf16.**

```
dtype histogram: {'U32': 498, 'F16': 1682}
```

Every other GDN model we serve is bf16. With f16 weights the activations get
promoted to fp32 (f16 ⊕ f32-scalar → f32, the `scalarLike` class), which the
mangled kernel name states outright — the input dtype list reads
`float float float bfloat16_t float bfloat16_t int32_t` for
q,k,v,g,beta,state_in,T. So q/k/v/beta are fp32 while `g` (our fused gate
kernel's output) and the state buffer stay bf16: a genuinely MIXED input set.

`transformer.zig` hardcoded `add_template_arg_dtype(config, "InT", .bfloat16)`
at five sites, and the blocked kernel body hand-declared
`const device InT* k_base = …` off it. fp32 buffer, bf16 pointer, compile
failure — and an MLX compile failure is not a Zig error, it aborts the process.
One request took the server down for every client.

The comment above the kernel had predicted the whole thing and then dismissed it:

> block size: MLX_SERVE_GDN_BLOCK_T (16|32|48, default 32 for bf16 — Metal's
> 32 KiB threadgroup limit governs; fp32 inputs would need 16, **but our GDN
> inputs are always bf16**).

**Why the stock kernel survived.** `GDN_KERNEL_SOURCE` indexes the raw buffer
names (`auto q_ = q + …`), so it adapts to whatever dtype arrives — exactly the
house rule the blocked port broke. `MLX_SERVE_GDN_BLOCKED=0` was therefore a
complete workaround, and that A/B is how the diagnosis was confirmed: same
model, same 501-token request, 22.6 → coherent generation with the kernel off,
`Unable to build metal library` with it on.

**Three things the fix had to get right, not one.**

1. `InT`/`StT` come from `mlx_array_dtype` of the actual arrays. `g`/`beta` need
   no template arg at all — they are read through raw names with an explicit
   `(float)` cast, which is why the mixed set never bothered them.
2. **Block size follows the dtype.** The staging arrays are declared *in* `InT`
   (`threadgroup InT k_s[TB][Dk+8]`), so fp32 doubles the footprint: at
   Dk=128/TB=32 that is 40,192 bytes against a 32 KiB threadgroup limit.
   `gdnBlockTFor(requested, dk, itemsize)` picks the largest supported TB that
   fits (fp32 ⇒ 16) and returns null — decline to the stock kernel — rather than
   ever dispatching over budget. Fixing the dtype alone would have traded a
   compile error for a different compile error.
3. **The store type is the OUTPUT's.** Making `InT` honest immediately broke the
   *stock* kernel too:

   ```
   error: assigning to 'bfloat16_t' (aka 'bfloat') from incompatible type 'float'
       y[dv_idx] = static_cast<InT>(out);
   ```

   Metal has no implicit float→bfloat conversion, and `y` is declared bf16 by us.
   `static_cast<InT>` had only ever compiled because InT happened to equal the
   output dtype. That is now `OutT`, set from the output declaration, in all
   three kernel sources.

Diagnosis order that worked: read the dtypes out of the mangled template name in
the error (they are all there), then the safetensors header histogram, then
`grep` for the hardcoded `.bfloat16` template args. The kernel name is the
fastest signal — it tells you what MLX actually saw, not what you assumed.

Guards: `gdnBlockTFor` unit test (pure, pins the staging arithmetic and the
clamp), plus fp32 and f16 cases in the blocked-parity sweep, which run through
the same `gdnBlockTFor` the production path does — so a regression that declines
the blocked route instead of clamping fails with `error.GdnBlockedDeclined`
rather than passing on the stock fallback.

### MLX's buffer pool is effectively unbounded, the clear cadence was skippable, and KV growth orphaned a whole cache every 256 tokens (#110)

Live 2026-07-27, reported against `ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve` driven from
Zed on a 128 GB Mac: the process climbed to **81.4 GB** in Activity Monitor while the
app's own panel read **GPU Memory 19.6 GB**, until the reporter killed it. Two
screenshots, taken at the same moment, ~61 GB apart.

Per this file's own diagnosis ladder — `active_bytes` NOT dropping after unload =
leaked HANDLES; active flat while phys climbs = allocator-CACHE growth — that gap is
MLX's buffer pool. Nothing was leaked; 61 GB was freed, parked, and never returned to
the OS. Three independent defects fed it.

**1. The pool has no useful bound.** `lib/mlx-src/mlx/backend/metal/allocator.cpp:63-64`:

```cpp
block_limit_ = std::min(1.5 * device_info()["max_recommended_working_set_size"], 0.95 * memsize);
gc_limit_    = 0.95 * device_info()["max_recommended_working_set_size"];
```

~121 GB and ~91 GB on a 128 GB Mac. MLX will not trim until the machine is already
dead. Our only real defense is explicit `mlx_clear_cache()`.

**2. Three decode paths never called it.** Auditing every `self.step` advance:

| path | advance | cleared? |
|---|---|---|
| `Generator.next` | +1 | yes, `step % 256` |
| `Generator.nextConstrained` | +1 | yes |
| `Generator.nextPld` | +1 / +num_emit | yes |
| **`Generator.nextDrafter`** | +1+m / +1+accepted | **no** |
| **`Generator.nextMtp`** | +1+m / +1+accepted | **no** |
| **`scheduler.runBatchedDecodeTick`** | +1 | **no** |

The reporter runs the `-mtp` checkpoint on a dense trunk, so `defaultEnableMtp` is on
and every one of his decode rounds took the one path that never cleared. `finishSlot`
didn't clear either, so what a turn stranded survived every later turn.

The `% 256` form is separately broken for exactly these paths: a spec round advances by
`1 + accepted`, so it can step clean over every multiple. Measured on the pure helper:
**at stride 5, zero clears in steps 0..1024.** A modulo cadence and a variable stride
are incompatible, and the failure is silent — a decode path that never clears is
output-identical to one that does.

**3. Linear KV growth made each strand enormous.** `updateDense`/`updateAffine` both
grew by a fixed `chunk_step = 256`: allocate a new capacity-sized buffer, copy, free the
old. MLX reuses a cached buffer only when `cached < size + 2*page_size`
(`buffer_cache.h:34`) — the next KV buffer is 512 KB past a 32 KB tolerance, so the
superseded buffer can never be reused for anything, least of all its own successor.

The arithmetic on his model: 64 layers, `full_attention_interval: 4` → 16 attention
layers, `head_dim: 256`, 4 KV heads, bf16 = **64 KiB/token**. At 89 K context that is
**~5.6 GB stranded every 256 generated tokens**. Eleven growth events ≈ 61 GB — the
observed gap, near exactly.

**Fixes, all three layers:**

- `server.mlxCacheLimitBytes` = `max(2 GB, min(8 GB, RAM/16))`, applied by
  `applyMlxCacheLimit()` ONCE at the top of `main()`. Not per serve path — this is the
  same shape as `runHeadlessServe` hand-rolling its own `ServerConfig` and silently
  eating the `--pld*` flags. `MLX_SERVE_CACHE_LIMIT` (bytes; `0` = leave MLX's default)
  is the A/B off-switch, and a tighter existing cap is never raised (media gen drops to
  1 GB on small-RAM machines, iOS boots at 384 MB).
- `Generator.advanceStep(n)` is the ONE place `step` and `completion_tokens` move, and it
  owns the clear. The predicate is `step -| last_clear >= interval` — interval
  arithmetic, which a variable stride cannot skip, and byte-identical to the modulo form
  for the stride-1 paths. `finishSlot` clears once more so a turn can't hand its
  transients to the next.
- `KVCache.nextCapacityPolicy` grows by +25%, floored at one chunk and capped at 8192
  tokens. Over a 100 K-token walk that is 27 growth events instead of 391, and the total
  superseded capacity drops from ~195× the final capacity to ~5.5×.
  `MLX_SERVE_KV_GROW=linear` restores the old policy.

The slack is honest, not hidden: `denseView` slices to `entry.offset` so attention never
sees it, and `prefix_cache.snapshotBytes` bills `mlx_array_size` (full capacity), so a
committed hot-cache entry is charged for what it actually pins. `truncate`,
`snapshot`/`restore` and the disk tier were all already capacity-agnostic (they work off
`offset` and explicit chunk positions) — audited, and worth keeping that way.

**Measurements.**

- Hermetic star test (`KV growth does not ratchet the MLX buffer pool`): 20,000
  single-token updates against a real 4-layer KVCache, sampling
  `mlx_get_cache_memory()`. **1544 MB → 274 MB.** Runs in ~3 seconds and needs no
  checkpoint — it reproduces the reporter's exact mechanism at toy scale.
- Integration (`tests/test_kv_cache_growth_memory.sh`, his model, 23 K prompt ×3 turns):
  `memory_mb - active_bytes` **14.61 GB → 3.87 GB** after one turn; footprint ratchet
  across turns 1→3 is 0.56 GB; pool peaks at 2.3 GB against the 8 GB cap. Red-on-revert
  verified against a rebuilt pre-fix binary.
- Pool sizing, measured rather than guessed: peak pool during a **66 K-token** turn on
  the 27B is **5.83 GB**, so the 8 GB cap has headroom at the context this class bites
  at.

**Perf.** The pool cap was the one plausible regression source, and a first pass at
65 K decode read −14%. That was **thermal contamination plus n=1**: the isolation runs
declined monotonically in temporal order (34.51 → 33.51 → 30.89 → 28.48) regardless of
arm. Re-run cooled and tightly alternated per boot, the same rung gives fix
33.76/34.90/29.96 vs old 30.56/32.10/33.35 — pair signs `+,+,−` with a spread far larger
than the mean gap. A full ABBA `bench.sh --family all` (34 mlx-serve cells, default vs
`MLX_SERVE_CACHE_LIMIT=0 MLX_SERVE_KV_GROW=linear`) lands at **decode −0.0%, prefill
+0.5%**. The absolute diff against `all-26.7.10.csv` looked like −12% to −25% across the
board — and the OLD arm measured in the same session showed the same depressed
absolutes, which is the whole reason cross-session absolutes are not admissible here.

**Visibility.** This was invisible for as long as `active_bytes` was the only memory
figure we served. `/props` now carries `memory.cache_bytes`, `/metrics` carries
`mlx_serve:mlx_cache_bytes` + `mlx_serve:mlx_active_bytes` beside the existing
`mlx_serve:memory_mb` (phys_footprint), and the tray renders `19.6 GB (+61 GB cache)`
once the pool passes 1 GB. Any future instance of this class names itself.

**Guards.** `KVCache growth strands bounded bytes` (pure, walks the policy and counts
events + superseded bytes), `KV growth does not ratchet the MLX buffer pool` (real MLX),
`clear cadence survives variable spec strides` (strides 1/3/5/9), `no decode path
advances `step` outside advanceStep` (source scan over `@embedFile("generate.zig")` +
`@embedFile("scheduler.zig")` — this is what stops the whack-a-mole; a new decode path
cannot reintroduce the hole), the `mlxCacheLimitBytes`/`mlxCacheLimitFromEnv` tables, and
`tests/test_kv_cache_growth_memory.sh`.

**Noted, not fixed:** `computeMemoryContext` (server.zig) bills `num_hidden_layers` as
full-attention layers. On `qwen3_5` only 1 in 4 are, so `kv_per_tok` is ~4× over-billed
and auto-context is correspondingly conservative (the reporter's "GPU-safe max 104K" on
a 256K model). Safe direction, real capability left on the table — separate change,
separate bench.

## Laguna XS 2.1 NVFP4: the decode gap was an f32 constant table (CLOSED)

Measured 2026-07-28 on an M4 Max (128 GB, ~546 GB/s) against the
Layr-Labs/mlxfast-challenge tree at `main` 55f965f — the same weights (SHA-verified
byte-identical to the challenge's pinned revision `841778bd`), the same silicon, the same
timed window (512-token prompt, 128 decode steps, temp 0, serial non-speculative).

**Measure the reference tree in the same window as your own, every time.** Mid-session the
box picked up concurrent GPU work from another process and every number inflated ~1.62x
(ours 40.5 → 65.6 ms/token). Re-running THEIR harness as a control showed 13.44 → 21.78 —
the same 1.62x — so the machine moved uniformly and the mlx-serve : mlxfast ratio was
**3.01 in both windows, identical to three significant figures**. Without the control run
the second window reads as a 60% regression in our tree. A cross-engine comparison is only
as trustworthy as the interleaved re-measurement of the thing you are comparing against.

| tree | decode ms/token | prefill tok/s |
|---|---|---|
| mlxfast **frontier** (every DARKBLOOM optimization on) | 13.442 | ~1463 |
| mlxfast **baseline** (every DARKBLOOM flag ablated off) | 17.173 | ~1460 |
| **mlx-serve**, as measured | 40.515 | ~1005 |
| **mlx-serve**, after the mscale dtype fix below | **13.384** | **~1335** |
| mlxfast frontier, control re-run in the SAME window as that fix | 13.099 | ~1451 |

Their entire published optimization catalogue is worth **+27.8%** (17.17 → 13.44).
mlx-serve started **2.36× slower than their UNOPTIMIZED baseline**, so porting their
fusions was never going to be the lever — the gap was in our forward pass, and it was one
line. The remaining gap after the fix is ~2% on decode and ~8% on prefill; their catalogue
(fused QKV, gate folded into `o_proj`, a fused routed `[gate; up]` bank, a certified
two-pass lm_head prune) is what that ~2% would have to come from, and none of it was worth
porting on top of a forward that was widening every weight read.

**Where the gap is not.** Each of these was A/B'd on the live decode and moved nothing:

- MLX's command-buffer batching (`MLX_MAX_MB_PER_BUFFER` / `MLX_MAX_OPS_PER_BUFFER`).
  Plausible on paper — MLX commits a buffer every 50 ops **or 50 MB of touched arrays**
  (`CommandEncoder::needs_commit`), and XS's bf16 attention weights are ~33 MB EACH, so we
  flush roughly per op. Sweeping 50 → 4096 MB: 40.9 / 40.5 / 40.9 / 41.8 ms. Flat.
- Metal residency (`mlx_set_wired_limit`; MLX's default is 0, i.e. nothing wired, and the
  routed expert banks are ~17.5 GB read at 8 random experts of 256 per layer). Wiring 24 GB
  and 48 GB: 40.1 / 40.1 / 40.4 ms. Flat.
- The MLX buffer-pool cap added in `c6413b4` (`MLX_SERVE_CACHE_LIMIT=0`). Flat.
- `MLX_SERVE_COMPILE_FORWARD=1` — compiles, but only the prefill chunk loop routes through
  the closure, so decode is untouched. Flat, as expected.

**Where the gap IS (RESOLVED 2026-07-28): a load-time f32 constant table silently
promoted the whole residual stream to float32.** Laguna's YaRN mscale vector
(`yarn_mscale`, `[head_dim]`, the reference's `cos/sin *= attention_factor`) was built
`.float32` at load and multiplied straight into post-RoPE q/k, which are bf16. mlx
promotes bf16 ⊕ f32 → f32, so:

```
q_rope * mscale(f32)  ->  q,k become f32
  -> SDPA runs f32     -> attn_out f32
  -> o_proj = mlx_matmul(f32 x, bf16 o_w)   -> o_w UPCAST on every token
  -> h + attn_out      -> the RESIDUAL is f32 from the first full-attn layer on
  -> every later q/k/v/o/gate, every routed expert, and lm_head upcast too
```

One line, and every weight read for the rest of the forward paid a widening pass.
`[dtype-trace]` says it plainly: residual **bfloat16** entering the layer loop,
**float32** leaving it. Fix = `constTableAs`, which hands the table back in the
activation's own dtype, cached per dtype (`DtypeCastCache`).

| | before | after |
|---|---|---|
| Laguna XS forward (`MLX_SERVE_DECODE_FWD_UBENCH`) | 41.2 ms | **14.6 ms** |
| Laguna XS decode, bench window | 40.585 ms/tok | **13.384 ms/tok** |
| Laguna XS prefill | 994 tok/s | **1335 tok/s** |
| Laguna S forward (also YaRN) | 23.2 ms | **17.7 ms** |
| lm_head alone | 4.6 ms | **1.2 ms** |

That is **3.03x** on decode, and it moves us from 2.36x slower than the mlxfast-challenge
tree's *unoptimized baseline* (17.173) to within 2.2% of its fully-optimized *frontier*
(13.384 vs a 13.099 control re-run interleaved in the same window). gemma-4-26B and
Qwen3.6-27B are unchanged — the path is laguna-only.

**Why every earlier hypothesis missed it.** The promotion is invisible to each thing you
would naturally reach for: op count is IDENTICAL (3226 with the bug vs 3080 in a clean
reconstruction), kernel choice is irrelevant (all three MoE decode kernels within 4%),
the KV cache is irrelevant (bypassing it entirely moved 0.4 ms), there is no host sync
(CPU graph build is 0.95 ms of a 41 ms token), and RAM was 94% free. It presents as a
*uniform ~3x* on attention, MoE and lm_head alike — which reads like a global platform
problem, not a one-line bug. The previous session's "dtype promotion ruled out, x/q_w/o_w
all bf16" was measured **before the layer loop**, which is exactly where it is still true.

**How it was actually found — bisect FORWARD, and let the reconstruction diverge.**
`diagProjBench` grew from a bare matmul loop into a rung ladder (`ProjRung`), each rung
adding one more piece of the real attention block, in both issue orders:

```
matmul_only  7.04 ms (405 GB/s)  442 ops     <- bare q/k/v/o
rope         7.38                 882
norms        7.88                1242        <- + input/post-attn norms + residuals
mlp         13.15                3080        <- + the REAL moeMLP: a whole layer
real_attn   36.29                3220        <- same layer, but the SHIPPED lagunaAttnWith
```

The reconstruction and the shipped function do the same number of ops on the same weights
and are 2.8x apart, so the bug had to be *inside* `lagunaAttnWith` and could only be a
difference the op count cannot see. Two rungs later (a `yarn_rope` rung proved the custom
YaRN freqs array was innocent, 13.17 vs 13.25) the only remaining difference was the
mscale multiply. **The rung that made the reconstruction disagree with the real function
is what localized it** — a reconstruction that merely matched would have proved nothing.

Probes kept, all env-gated: `MLX_SERVE_DECODE_FWD_UBENCH=N` (forward-only timing, split
into CPU graph-build vs GPU eval, ops/forward, and a sound lm_head ablation — lm_head is
terminal so removing it cannot change upstream work), the `diagProjBench` rung ladder it
fires, `MLX_SERVE_LAYER_CAP=N` (layer-count sweep: Laguna XS is dead linear at
**0.896 ms/layer + 5.2 ms fixed**, which is how "outside the loop" was priced), and
`MLX_SERVE_LAGUNA_UBENCH=1`.

**Still ruled out, each by live A/B — do not re-litigate:** command-buffer batching
(`MLX_MAX_MB_PER_BUFFER` 50→4096, `MLX_MAX_OPS_PER_BUFFER` 50→1000); Metal residency
(`mlx_set_wired_limit` 0→48 GB); the buffer-pool cap from `c6413b4`
(`MLX_SERVE_CACHE_LIMIT=0`); `MLX_SERVE_COMPILE_FORWARD=1` (only the prefill chunk loop
routes through the closure); graph-scheduler lookahead (`MLX_BFS_MAX_WIDTH` 20→4000); a
lazy load-time transpose re-materialized per step; 2-D vs 3-D activations; dispatch count
(trivial dependent ops are 1.7–2.4 µs each, ~2 ms of the token).

**This fix is a correctness fix, not a speed/accuracy trade.** It is not bit-identical to
the pre-fix binary, and that is the point: the f32 product was the DEVIATION. The stock
MLX path for a bf16 Laguna is `rope_input_with_mscale<bfloat16, true>`, which applies
`float(bfloat(x * bfloat(mscale)))` — mscale cast to bf16, product rounded to bf16 — and
the mlxfast-challenge tree's hand-written kernel reproduces exactly that
(`bfloat rounded_mscale = bfloat(yarn_mscale)`, then
`float(bfloat(normalized[i] * rounded_mscale))`, `LagunaRuntimeModel.swift`). Casting the
table to the activation dtype puts us ON that path. Running attention in accidental f32
was never "safer": this is a quantized checkpoint calibrated against bf16 arithmetic, the
same trap as the MageFlow Turbo rule (f32 is not "more accurate", it washes). Nothing in
the repo pins Laguna token bytes, so there was no baseline to re-cut.

**The class, and the guard.** A constant table built in one dtype at load and multiplied
into an activation in another silently widens everything downstream of it. The two other
f32 tables on this decode path already got it right and are the pattern to copy: the MoE
router computes in f32 deliberately and `astype`s back to bf16 before returning, and
M-RoPE's cos/sin are built f32 then cast to bf16 before they touch q/k. Same family as
MageFlow's `scalarLike` rule. Guards: `constTableAs hands a load-time f32 table back in
the activation dtype`, `a constant table must not widen the activation it scales`, and a
source scan (`the YaRN mscale table never reaches a multiply in its load-time dtype`) —
the scan needles are assembled at comptime, because spelled whole they appear in the
test's own source and the scan satisfies itself (it did, on the first attempt, and passed
green against a reverted fix).

### The intra-step async-eval ladder is a null lever for us (measured, not assumed)

`DARKBLOOM_DECODE_ASYNC_STAGE` is worth +9.7% in their tree: fire `asyncEval` at layer
boundaries so the GPU starts while the CPU is still building the graph. Ported as
`Transformer.ladderStep` / `MLX_SERVE_DECODE_ASYNC_LADDER` and swept on Laguna XS:

```
off 39.88 | stride 8 40.22 | stride 4 40.37 | stride 2 41.08 | stride 1 41.49 ms/token
```

Monotonically WORSE as the stride tightens. The reason is structural: their worker cannot
pipeline BETWEEN steps (its protocol is one token per request), so the ladder is the only
overlap it can buy; mlx-serve already overlaps at the step boundary (`generate.zig`'s
build-next → `mlx_async_eval` → resolve-pending), so the overlap is already collected and
the ladder only adds submissions. It also confirms we are **not** CPU-graph-build-bound —
if we were, the ladder would have paid.

Shipped **default OFF** with the sweep recorded in the doc comment; `"auto"`/`"<n>"` opt in
for a future arch whose CPU side is the bottleneck. Output is bit-identical either way
(verified: 160 greedy tokens, temp 0, byte-for-byte). **Rule**: a lever that pays in
another engine's harness may be paying for a constraint that engine has and we do not —
port it, measure it, and let the number decide the default.

## The f16 checkpoint whose per-channel tables turned the whole residual f32

**2026-07-29, `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, worth 14.8%.**

The Laguna YaRN-mscale bug (above) was found by luck: the MoE forward happened to carry a
`[dtype-trace]` log and the other five forward paths had none. So the follow-up was to
generalize the trace and sweep every arch on disk — `DtypeTrace` in `src/mlx.zig`, armed by
all five `forward*With` paths plus batched-decode, the diffusion decoder and both vision
towers, with a class guard (`every transformer forward path is watched by a dtype trace`)
that keys on the forward-path SIGNATURE rather than on the shape of its layer loop. That
distinction is load-bearing: BERT iterates its layer slice directly while every other path
counts to `num_hidden_layers`, and a rule that keyed on the loop form would have skipped it
silently.

The trace logs both endpoints and, in between, only the layers where the dtype actually
MOVES — so a mid-stack widening names one layer instead of drowning in forty identical
lines, and after the first forward the whole thing costs nothing.

One arch out of nine widened:

```
[dtype-trace] moe: residual in = bfloat16, first weight = n/a
[dtype-trace] moe: residual widened at layer 0: bfloat16 -> float32
[dtype-trace] moe: residual out = float32
```

The checkpoint is F16 throughout — despite `config.text_config.dtype: "bfloat16"`, which is
simply wrong (read the CHECKPOINT, not the config: the Kokoro `AdaIN1d` rule again).
`loadSafetensorsFile` already narrowed f16 `.scales`/`.biases` to bf16 (the hy_v3 mixed-dtype
gather story), but deliberately left plain weights alone — including every NORM weight. So
`rmsNorm(bf16 residual, f16 weight)` promoted at layer 0 and the residual stayed f32 for all
64 layers, upcasting every weight read after it. Same class as the mscale bug, one level up:
there it was one constant table, here it is a whole category of per-channel table.

Fix: extend the load-site rule to any 1-D f16 tensor (`narrowsLoadedF16`). A 1-D tensor is a
per-channel table — norm weight, bias, `A_log`, `dt_bias` — that is multiplied or added
straight into the residual. Multi-dimensional dense f16 weights are left alone: those are
matmul OPERANDS, MLX picks its kernel off that dtype, and narrowing one is a
kernel-selection change rather than a promotion fix. Measured as a wash (23.15 vs 23.47 ms,
inside boot drift), so the minimal rule shipped.

**27.99 → 23.88 ms/forward, 14.8%**, greedy output byte-identical at temp 0 over 160 tokens,
and a strict no-op on every bf16 checkpoint (`[dtype] narrowed 0` — Laguna, gemma4-26B-A4B
and Qwen3.6-35B-A3B all unchanged). Kill switch `MLX_SERVE_F16_NARROW_1D=0`.

**The measurement nearly went the other way, twice.** The first paired readings said the fix
made the model 2x slower (51.3 ms) and then 5x slower (136 ms, with lm_head at 72.5 ms
against 1.4 ms). Both were garbage — first-boot artifacts from loading an 8.4 GB checkpoint
back-to-back with the previous one still in the page cache. Re-running the same two configs
reversed the sign completely (23.5 vs 27.6). Then a four-boot alternating sweep drifted
monotonically (23.5, 23.2, 32.0, 34.6) with no relation to the mode — thermal soak. Only a
paired A/B with a 45 s cooldown between boots gave a stable answer, three rounds with no
overlap. **A model this size needs a settle window between boots; the first boot after
another large load is not a measurement.** Two separate wrong conclusions were one run away
from being written down as fact.

Two things found on the way and left explicitly open:

- The same checkpoint's Qwen3-VL vision tower runs f32 end-to-end (`residual in = float32,
  first weight = float16`) because its weights are 2-D f16. Vision is a one-shot prefill
  cost, and narrowing matmul operands is a kernel-selection change that needs its own A/B —
  not folded into this rule.
- `diagProjBench` reshapes q back through `num_attention_heads * head_dim` and Qwen3.5-2B
  ships a q_proj of 4096 against a computed 2048, so the diagnostic raised an MLX reshape
  error — an uncatchable process kill — and one probe boot took the server down.
  `projLadderFits` now declines with a log, because a silent skip is indistinguishable from
  "this model has no dense attention".

**Archs still unswept** (not on this disk): `qwen3`, `qwen3_next`, `nemotron_h`, `lfm2`,
`hy_v3`, `diffusion_gemma`, and the media engines. The trace is armed on every path they
would take, so booting one is now the whole check.

## What a 3x forward fix invalidated: two Laguna defaults and a spec gate

**2026-07-29.** Fixing the YaRN mscale promotion did not just make Laguna faster — it
moved every ratio that had been calibrated against the slow forward. Three decisions were
re-measured on the fixed tree and two of them flipped.

### The KV half was real, but it pays LESS at long context, not more

`updateDense` takes the cache dtype from `new_k`, so while the residual was f32 the
full-attention KV was STORED in f32. Laguna XS has 10 full-attention layers of 40 (the
other 30 cap at a 512-token sliding window), 8 KV heads x 128 head_dim.

Paired ladders in one window (serial, `--no-pld`, prompt caches off, one boot per lane,
the fix reverted at its single `constTableAs` call site for the pre lane):

| ctx | pre ms/tok | post ms/tok | speedup | pre prefill | post prefill |
|---|---|---|---|---|---|
| 512 | 39.89 | 13.29 | 3.00x | 907 | 1232 |
| 4096 | 41.84 | 13.83 | 3.02x | 903 | 1232 |
| 16384 | 43.35 | 14.93 | 2.90x | 382 | 637 |
| 32768 | 48.64 | 17.66 | 2.75x | 253 | 403 |
| 65536 | 52.85 | 20.27 | 2.61x | 141 | 233 |

Subtract the 512-token base and the KV term falls out exactly: at 32K it is 8.75 ms before
and 4.37 ms after — **2.00x**, precisely the f32→bf16 halving, and 4.37 ms for 1.34 GB is
307 GB/s, a sane rate. The mechanism is confirmed to two significant figures.

The *prediction* that it would therefore be worth MORE than 3.03x at 32K is wrong, and
wrong for a reason worth keeping: the forward term shrinks 3x while KV only shrinks 2x, so
KV becomes a larger share of a smaller token and the blended ratio FALLS with context
(3.00x → 2.61x). A fix that improves two terms by different factors does not compound.

### gatherQmv lost to stock `gather_qmm` on both Laguna models

`batchedExpertDecodePolicy` opted Laguna into the batched/gatherQmv expert path by default
on a measured 17→48 tok/s. That measurement was taken while the forward was ~3x inflated.
Re-measured on the fixed tree, alternating boots, 512-token window, serial:

| model | gatherQmv | stock gather | batched take+qmm |
|---|---|---|---|
| Laguna XS | 13.296 ms/tok | **13.113** (+1.4%) | 14.839 |
| Laguna S | 17.02 ms/tok | **15.45** (+9.2%) | — |

Every round, no overlap. The reason is the one already written down for gemma4-26B-A4B:
these kernels beat MLX only where the O(EXPERT-BANK-SIZE) addressing term DOMINATES the
expert math. Making the rest of the token 3x cheaper made the expert math a much larger
share and the bank term a smaller one, so MLX's better-tuned `gather_qmv_fast` tile wins.
No arch opts in by default any more; both alternatives stay one env var away, and the next
arch with a genuinely bank-dominated decode gets measured back in. Laguna XS forward:
14.203 → 13.447 ms, ops/forward 3226 → 2329.

### The PLD yield gate was warming up 4x too long

The yield gate exists because PLD's cold path (no n-gram match) runs an UNPIPELINED forward
plus a synchronous host read of the sampled token, against `next()`'s async-pipelined step.
Its 32-step warmup was calibrated when the AR step cost 3x more; the same absolute tax is
now a ~3x larger share.

Swept on Laguna XS — one boot, serial vs unconstrained alternating per REQUEST, timings
from the server's own `timings` object, 5 runs median. The matrix deliberately includes
PLD's WIN case: an earlier sweep over only loss cases drove the warmup toward zero, which
would have thrown the win away.

| warmup | echo-edit | code-edit | free-form | explain | qa |
|---|---|---|---|---|---|
| 32 (was) | +76.9% | -4.3% | -1.2% | -1.4% | -7.2% |
| 16 | +70.1% | -3.4% | -0.4% | -0.3% | -0.8% |
| **8 (now)** | **+77.4%** | **+0.3%** | **-0.2%** | **-0.3%** | **-1.5%** |
| 4 | +76.6% | +3.9% | -0.2% | -0.1% | -1.7% |

8 recovers essentially the whole loss with the +77% untouched. 4 measured no worse, but a
gate deciding on four observations is fitting noise, and a short preamble before a file
echo is the NORMAL agent shape — exactly what a too-eager trip punishes. A premature trip
is bounded anyway (`specShouldReenable` re-checks every 32 steps). `SPEC_YIELD_WARMUP`
sweeps it without a rebuild.

Note the shape of the loss: `free-form` and `explain` emit NO `[spec-stats]` line at all —
PLD never engages — and were still 1.2-1.4% slower. Enabling speculation costs something
even when the prompt gate declines it, because the request still routes through the
unpipelined step function.

### Method notes from this round

- **Single runs lied three times, in both directions.** A "clean re-run" of gemma4-31b
  produced anomalously HIGH values that got merged into the baseline, so the next run read
  as a -14% regression; two repeats put it back at baseline. `qwen36-27b/mtp/code` read
  74.3 against a 75 floor and then 77.3. `gemma4-31b/*/prefill` is BIMODAL (~178-191 vs
  ~205) across the whole session. The baseline is now per-cell MEDIANS over repeated runs
  with the spread recorded, because a one-run baseline makes every later diff a coin flip.
- **Attribute before believing.** Every flagged cell this round was structurally
  unreachable from the changes (gemma4 prefill cannot be touched by a PLD-decode gate, a
  no-op f16 narrowing, or a gather policy that was already `false` for gemma4). Check
  reachability first; it is faster than another bench run and it is what makes the repeat
  a confirmation rather than a hope.
- **`MLX_SERVE_DECODE_PROFILE` is not a sizing tool.** It forces an eval at every phase
  boundary, which destroys pipelining: 47.4 ms/token against a real 13.1, with the router
  phase absorbing sync cost and reading LARGER than the experts it gates. Use the ladder or
  the real forward.
- **The `diagProjBench` ladder still has a constant input**, so its `mlp` rung reads a
  resident expert bank and is a LOWER bound (4.63 ms, 34% of the token). Feeding it eight
  varying inputs was tried and reverted: the lazy `astype` nodes outlive their f32 sources
  in a way the single-input form never exposed and the rungs SIGBUS (status 138). It is a
  diagnostic; the real forward is the honest number.
- **A ladder guard must use the same geometry the ladder does.** `projLadderFits` first
  compared against a global `num_attention_heads * head_dim`, but the rung uses
  `cfg.layerNumHeads(li)` — Laguna runs 48 heads on full-attention layers and 72 on sliding
  ones, so the global form declined 30 of 40 layers and silently under-reported the
  ladder's bytes by 5x (85 GB/s instead of 412).

## Fusing decode dispatches: only the critical path pays (2026-07-29)

Round on top of the 26.7.12 tree, on Laguna XS 2.1 NVFP4 (M4 Max, 20 GB checkpoint,
`tests/fwd_ubench.sh`, 20 decode-width forwards per boot, paired per boot).

### Pricing a dispatch: `MLX_SERVE_DISPATCH_PROBE=N`

Before writing any fusion, the probe injects N extra elementwise kernels per MoE layer
inside `moeMLP2` — a multiply by an exact 1.0, which is output-identical for every finite
value and, crucially, FEEDS the expert path so MLX cannot elide it. That is what makes it
sound where the obvious "run the component twice and read the marginal cost" is not: it is
pure (no KV writes) and observable (the result is consumed).

    probe   0        4        8
    run A   13.083   13.371   13.630 ms/forward
    run B   13.131   13.305   13.657

Slope ≈ 0.067 ms per dispatch-per-layer, i.e. **~1.5–1.7 us per GPU dispatch** at 40
layers. Handy, and misleading if used alone — see the null result below.

### What that price does NOT buy: the attention output gate

Laguna's per-head `softplus(g_proj(x))` gate is four dispatches a layer (cast to f32,
logaddexp, cast back, broadcast multiply). `fusedAttnGate` replaces all four with one
kernel that is bit-identical to the chain (MLX's `LogAddExp` text, same rounding points,
pinned by test). By the probe slope that is worth ~0.2 ms. Measured, three pairs:

    off  13.112  13.098  13.030
    on   13.117  13.116  13.160

Nothing — slightly negative. **The gate chain does not sit on the critical path**: it
depends only on the layer input, so the GPU already overlaps it with the q/k/v projections
and SDPA, and deleting overlapped work buys zero wall clock. The probe measures dispatches
inserted INTO the dependency chain, so its slope is an upper bound that only materialises
for fusions that actually shorten that chain. Ships default OFF
(`MLX_SERVE_ATTN_GATE_FUSED=1` opts in).

A first version of the same kernel was worse still (+2.3%): it staged the per-head gate in
threadgroup memory and had ONE 256-thread group walk all 6144 activations of a row. At
decode that is a single GPU core doing what the elementwise multiply it replaced spread
over the whole device. Recomputing softplus once per element — 10 ALU ops on a cached
value — and shaping the launch as one thread per output element got it back to neutral.
A fused kernel inherits the parallel shape you give it, not the one the ops had.

### What it does buy: the MoE router, and the SwiGLU

Both sit on the chain (router → expert gather; gate/up → activation → down_proj), and both
paid, byte-identical, on three consecutive pairs each:

    MLX_SERVE_MOE_ROUTER_FUSED   off 13.315 13.322 13.411 | on 13.113 12.934 13.395
    MLX_SERVE_SWIGLU_FUSED       off 13.204 13.106 13.180 | on 13.012 12.974 12.978

Together, position-balanced across six pairs in both orders, medians 13.418 → 13.061,
**−2.7%**, with `/v1/chat/completions` at temp 0 byte-identical over 260 tokens on Laguna
XS, gemma4-26B-A4B (the softmax router arm) and dense Qwen3.6-27B (the SwiGLU arm).

### Reproducing MLX's arithmetic is not the same as reproducing MLX's formula

The router kernel's first version computed "the same" softmax in one clean f32 pass. It
diverged from the chain on gemma4-26B-A4B at token ~80 — the known INT4 near-tie argmax
class, triggered by an avoidable numerical change. Making it bit-equal took three separate
corrections, each found by asserting BIT equality with the chain in a unit test:

1. **`fast::exp`, and a multiply by the RECIPROCAL.** `softmax_single_row` divides once,
   into a reciprocal, and uses `fast::exp` (its own comment says softmax does not need the
   precise one). Both matter after the rounding to bf16.
2. **The reduction TREE, not just the reduction.** MLX launches `ceil(E/4)` threads, each
   summing four CONSECUTIVE elements in order, then one `simd_sum` per simdgroup, then a
   final `simd_sum` over the per-simdgroup partials in lanes 0..S-1 with zeros above.
   A strided-by-32 sum over the same 256 values is a different float and was 1 ulp off.
   The kernel emulates those virtual threads 32 at a time so the butterfly is identical.
3. **`sum` over the top-K accumulates in the INPUT's dtype.** `reduce.metal` instantiates
   bfloat16 sums as `(bfloat16_t, bfloat16_t)` — U is bf16, not f32 — and `row_reduce_small`
   gives an 8-wide row to ONE thread, folding it in ascending order. A simd butterfly in
   f32 over the same eight probabilities is off by 2 ulps.

Same discipline for the sigmoid arm: MLX's `Sigmoid` is the two-sided
`y = 1/(1+exp(|x|)); (x<0) ? y : 1-y`, not `1/(1+exp(-x))`, and the hy3 chain casts to f32
FIRST so that arm is genuinely f32.

### A JIT custom kernel and MLX's metallib do not agree on transcendentals

The fused SwiGLU replaces `sigmoid + multiply + multiply`. Writing MLX's `Sigmoid` source
verbatim, on the same `T`, still diverged live at token ~55 while a random-sampling unit
test said bit-identical. Cause: MLX's kernels ship in a metallib built with one Metal math
mode; `metal_kernel` JITs custom sources with `CompileOptions{math_mode = Safe}` and mlx-c
exposes no way to ask for another. So `1 / (1 + exp(|x|))` lands a rounding apart.

An exhaustive sweep of all 65536 bf16 patterns found it on **exactly one** value
(-6.84375, ref -0.0072631836 vs -0.0073242188). One in 65536 sounds unreachable; an
8192-wide MLP draws 8192 values from that domain every layer, so a 260-token greedy run
hits it in the first few dozen tokens. **A 16-bit activation has only 65536 possible
inputs — sweep them all instead of sampling.** Random values over `[-5, 5]` plus a few
extremes passed every time.

The fix removes the transcendental instead of chasing it: `mlx_sigmoid` is evaluated over
every 16-bit pattern once at first use and the kernel indexes the result by the input's
bit pattern (`swigluSigTable`, 128 KB resident, built with MLX's own op so it is exact by
construction). The two multiplies need no such care — a bf16/f16 product is exact in
float, so one rounding is one rounding on any compiler. f32 is DECLINED outright rather
than shipped exact-for-some-dtypes.

Also rejected on the way: extending `mlx_compile` to the silu arm of `compileGeglu`. It is
one line, gets the same 2.4%, and is NOT output-preserving — a same-binary greedy A/B
diverged at token ~55. mlx_compile's elementwise fusion does not reproduce the chain's
intermediate rounding.

### Two other numbers from the round

- **`gatherQmv` rebuilds its `mlx_fast_metal_kernel_config` on every call** — 3 per MoE
  layer, ~10 FFI calls each — which shows up as `ops/forward` 3226 vs 2329 and **+0.4 ms
  of CPU graph build per token**. Its GPU eval is 12.73 ms against stock `gather_qmm`'s
  12.47, so it loses on both counts today, but roughly 60% of the gap it is charged is
  host-side config construction, not kernel time. The new kernels cache their config keyed
  by geometry; anything reaching for `metal_kernel` in a per-layer loop should.
- **Layer-cap refit on the fixed tree** (`MLX_SERVE_LAYER_CAP`, N = 10/20/30/40 →
  4.345/7.293/9.779/12.743 ms GPU): Laguna XS is linear at **0.280 ms/layer + 1.55 ms
  fixed**, and ~0.72 ms of that fixed part is lm_head. The rest is the probe's own
  per-forward `mlx_array_eval` sync, which a real decode loop pipelines away.

## The MoE expert path: fusing gate+up flipped a losing kernel into the default (2026-07-29)

Second half of the same round. The expert path was the biggest addressable block left —
Laguna XS reads ~630 MB of expert weights per token and was doing it at roughly 175 GB/s
against 412 GB/s for the dense attention projections in the same forward.

### The kernel was never the problem; the SHAPE of the work was

`gatherQmv` (our in-place gather-qmv, which reads the expert bank without materialising
the top-K) had been demoted earlier in 26.7.12 because it lost to stock `gather_qmm`. Two
separate things were being charged to it:

- **+0.4 ms per token of CPU graph build.** It rebuilt its `mlx_fast_metal_kernel_config`
  on every call — three calls per MoE layer, ~10 FFI calls each — which shows up as
  `ops/forward` 3226 vs 2329. That is roughly 60% of the margin it was losing by, and none
  of it is kernel time. Now cached, keyed by geometry.
- **Three dispatches where one would do.** gate gather -> up gather -> activation, all
  strictly serial into `down_proj`.

`gatherQmvGateUp` computes BOTH dot products in the same simdgroup — the x element loaded
for the gate FMA is reused for the up FMA, the two dequant chains interleave instead of
running as separate launches, and the SwiGLU is applied before the write. It halves the
simdgroup count rather than the work (4096 over 40 cores was ~100 per core, so there was
occupancy to spend on ILP).

    Laguna XS, ms/forward     stock 13.007 / 13.568   fused-qmv 12.768 / 13.114
    (GPU eval)                      12.136 / 12.569             11.854 / 12.051

That is the same kernel family that lost by 2% before the fusion winning by 2-3% after.

### The predicate is "would the fused kernel engage", not a model_type list

Split `gatherQmv` lost on Laguna AND on gemma4-26B-A4B, so no arch opted in. Rather than
add laguna back to a list, `useGatherQmvDecode` gates on exactly the conditions the fused
kernel needs — silu activation (it is baked into the kernel) and matching quant geometry
on gate and up. gemma4's gelu MoE therefore keeps stock gather and cannot regress, with no
arch name anywhere in the predicate. Validated on the two archs the predicate newly opts
in:

    Qwen3.6-35B-A3B (qwen3_5_moe, affine 4-bit gs64)
      GPU eval  stock 7.489 / 7.558   fused 7.390 / 7.306     (total: neutral to -1.6%)

Not bit-identical to stock `gather_qmm` — that is the sanctioned qmv-vs-qmm reduction-order
class, and the evidence chain is transitive: a new test pins fused == split `gatherQmv`
bit-for-bit, and the existing tests pin split `gatherQmv` no-worse-than-stock against fp32
dequant ground truth. Coherence spot-checked live on Qwen3.6-35B-A3B.

### Cumulative, measured properly

Laguna XS decode forward, whole round in ONE window, four pairs alternating which arm
boots first:

    base  13.331  13.348  13.204  13.344   median 13.338
    new   12.823  12.787  12.762  12.778   median 12.783   = -4.2%

All four pairs favour the new tree and each arm's spread is under 0.5%.

An earlier write-up of this round quoted **-5.8%**, obtained by chaining a baseline median
from the morning's window to a post-change median from the afternoon's. That is the trap
this very file warns about one section up, committed while documenting the round that
found it. A cumulative figure has to come from one window with both arms interleaved, or
it is arithmetic on two different machines. The per-lever numbers were each paired inside
one window and are unaffected.


## A second null: fused residual + RMSNorm (2026-07-29)

mlxfast item 5's norm tail. `h = h + branch; normed = rms_norm(h, w)` is two dispatches and,
unlike the attention gate, they really are on the critical path — so by the round's first
rule it should have paid. It did not: six pairs in both orders (on a machine under other
GPU load, so read the RATIO only) gave ON median 14.890 vs OFF 14.747 ms/forward, 4 of 6
pairs favouring OFF. Ships DEFAULT OFF, `MLX_SERVE_ADD_RMSNORM_FUSED=1` opts in; the
kernel is bit-identical to `mlx_add` + `mlx_fast_rms_norm` (all four widths x both dtypes,
including the ragged tail and a non-power-of-2 axis) and stays for future use.

The reason completes the rule. **Being on the critical path is necessary, not sufficient —
the fusion also has to remove real work.** Here it barely does: `add` is 2 reads + 1 write
of a 4 KB row, `rms_norm` is 1 read + 1 write, and the fused kernel still needs 2 reads +
2 writes because the residual sum is required downstream and must be emitted as a second
output. Net saving: one 4 KB read, in exchange for replacing MLX's metallib-compiled
`rms_single_row` with a JIT copy compiled under a different math mode.

Contrast the three that paid, each of which replaced something substantial rather than a
launch: the router killed a 256-wide `argpartition` sort, the SwiGLU killed a three-op
activation chain, and `gatherQmvGateUp` merged two whole GEMV passes over the expert bank.

The port also has a reusable piece: replicating `rms_single_row` exactly means keeping
MLX's launch geometry (ceil(axis/4) threads, four CONSECUTIVE elements each, `simd_sum` per
simdgroup, a zero-initialised `local_sums` plane, one final `simd_sum` over the partials)
so the reduction is the same TREE, plus `metal::precise::rsqrt` — upstream uses the precise
variant precisely so this is reproducible. Anything past `RMS_LOOPED_LIMIT` (4096) switches
MLX to a differently-shaped kernel and is declined rather than guessed at.

## Item 1, fused QKV: a null at BOTH widths, and two layout traps on the way (2026-07-29)

mlxfast item 1, and the one the handoff doc guessed our prefill gap was. Implemented as a
WEIGHT-LAYOUT fusion rather than a kernel: concatenate q/k/v along their output axis at
first use, one matmul, three slices (`buildFusedQkv` / `sliceQkvPart`,
`MLX_SERVE_FUSED_QKV=1`). The norm half of `lagunaFusedNormQKVProjection` was deliberately
left out — `fusedAddRmsNorm` had already shown that a 4 KB norm is not real work next to a
33 MB weight read.

**Decode** (Laguna XS, four pairs, both orders): on 12.924 / 12.890 / 12.733 / 12.669
against off 12.750 / 12.979 / 13.023 / 12.808 — medians 12.812 vs 12.879, one pair
favouring off and three favouring on, all inside each arm's own spread. **Prefill**
(`tests/prefill_ab.sh`, ~7 900-token prompts, server-reported `prompt_ms`, nonce-defeated
cache): on 8856 / 8993 / 9401 / 9157 / 9481 / 9202, off 9373 / 9773 / 9678 / 8866 / 8909 /
9128 — the two blocks CONTRADICT each other (on wins 4% in the first, off wins 3% in the
second). Both null.

Decode is the attention-gate story again: q, k and v are independent, so the GPU was
already overlapping their launches. Prefill is the more interesting miss — the "one pass
over x instead of three" argument is real (32 MB read once instead of three times) but at
M≈7900 each projection is already a compute-saturating GEMM, so the saved activation reads
hide behind the math. Ships opt-in, and it would stay opt-in even if it had won: the
concatenated copy is ADDITIVE (the originals stay live for other paths), +33.6 MB per layer
= ~1.34 GB on Laguna XS.

### Trap 1: concatenating pre-transposed weights silently changes which kernel runs

Dense weights are lazy TRANSPOSE VIEWS of row-major `[out, in]` buffers
(`maybeTransposeForBf16`), and mlx's matmul recognises that and dispatches the gemv that
walks each output's row contiguously. Concatenating the VIEWS on axis 1 and materialising
produces a genuinely row-major `[in, out_total]` matrix — a different kernel with a
different accumulation order. It passed a small unit test and diverged on the live model at
byte 164 of a 260-token greedy run. The fix is to transpose each back to `[out, in]`, join
on the OUTPUT axis, materialise, and hand back a transposed view of THAT: row j of the
joined buffer is then byte-identical to row j of the original and the same kernel reads it.
Divergence moved from byte 164 to byte 695.

### Trap 2: same layout is still not bit-identical, because mlx picks kernels by SHAPE

Even with the layout right it still diverges eventually — the concatenated output is 8192
wide where q alone was 6144, and mlx's gemv heuristics (split-K and tiling) key on that, so
the reduction order can change. This is the sanctioned qmv-vs-qmm class, not a bug, but it
means **a weight-layout fusion cannot be assumed output-preserving just because the maths
is identical.** Prove it per shape or treat it as a numerics change.

### And the test that should have caught trap 1

The first unit test used a 256-wide contraction dim and passed both traps. Over 256 terms
the two accumulation orders happened to agree in bf16; at 2048 they do not. A parity test
for a reduction-order bug needs a contraction dim in the same ballpark as the real one —
small shapes make the bug invisible, not smaller.

## Decode-only dense-attention requant (`--decode-attn-quant`, 2026-07-29)

Ported from the mlxfast-challenge tree's biggest single lever (their "native affine"
DARKBLOOM_NATIVE_AFFINE_QKV / _OPROJ stack, which took their frontier from ~1.12 to
~1.385 in two days of band-limited ratchets). The observation: Laguna ships BF16
attention over NVFP4 experts, and at decode the four attention projections are ~3 GB of
a ~4 GB per-token weight read. An INT8 group-32 affine side copy halves that traffic.

Implementation (`transformer.zig`): `attnDqFor` builds the side copy lazily on the first
eligible dispatch, keyed by the weight's ctx pointer (a probed-ineligible weight caches
its refusal); `buildAttnDqCopy` transposes the loaded [in, out] dense weight back to
[out, in], materialises, quantizes int8 g32 affine, and evals eagerly so the build never
rides a token's graph. The dispatch hook is `attnProj`, called from `lagunaAttnWith` and
`gatedFullAttnWith` with `batch == 1 and !is_prefill`.

Measured (Laguna XS, fwd_ubench, 3 pairs alternating boot order): 13.285 -> 10.205
ms/forward median, -23.2%, all pairs in both orders. Quality characterization: six greedy
code/reasoning prompts on/off — two byte-identical, four diverged in WORDING while
reaching the same correct answers (the quantized arm's Zig snippet was the more correct
of the two). Shipped default ON per that characterization; `--no-decode-attn-quant`,
the app Settings toggle, or `MLX_SERVE_DECODE_ATTN_QUANT=0` restore exact dense decode.

The three load-bearing subtleties:

1. **Spec-verify must see the SAME weights as decode.** The gate is `!is_prefill`, not
   `seq_len == 1`: a verify forward that read the dense weights would score drafts
   against a different distribution than the decode steps that produced them, making
   PLD-on vs PLD-off diverge at temp 0 (the spec-equivalence invariant). With verify
   quantized too, draft/verify/AR are all consistent and prefill stays the exact anchor.
2. **Prefill stays dense deliberately.** It is compute-bound (no perf win available) and
   it writes the KV that the whole conversation attends to — keeping it exact bounds the
   compounding. Decode-written KV positions do carry the perturbation, which also means
   a disk-cache entry written by an ON boot restored into an OFF boot differs from what
   that boot would compute — same order of effect as the int8 step itself, noted not tagged.
3. **The side copies are additive memory** (~9/16 of the dense attention bytes; ~1.7 GB
   on Laguna XS) and live until the model unloads.

Not attempted yet from the mlxfast stack: their NVFP4-g16 tail-layer variant (layers >=32
of 40 quantize to 4-bit — per-layer amplification is ~15x lower late; worth another ~6% of
the token if the band holds) and o_proj/QKV single-layer depth probes. The toggle name
stays format-agnostic so those can land behind it.

## Fused decode QK-norm+RoPE (2026-07-29, mlxfast 9e06de6 class)

At decode the q/k chain (reshape → per-head RMSNorm → transpose → RoPE → YaRN mscale on
full layers) is strictly serial ahead of SDPA — nothing overlaps it — and costs 4-6
dispatches per layer for a few KB of work. `fusedQkNormRope` does q AND k in one
32-thread-per-head dispatch. mlxfast promoted the same fusion at +1.73%, their largest
single bit-exact win, AFTER first rejecting it at -0.19% — the regression was one
redundant `threadgroup` broadcast + barrier for a value `simd_sum` already hands every
lane; our kernel never had the barrier.

The three things that made it bit-identical (pinned by the `qk norm rope fused` tests,
max_diff exactly 0.0 across offsets and both laguna geometries):

1. **cos/sin extracted, never re-derived**: a [1,1,1,rd] f32 probe (ones | zeros) through
   the stock `mlx_fast_rope` at the token's offset rotates to exactly [cos | sin] — the
   same floats the composed kernel uses, because the angle math inside rope is float
   regardless of tensor dtype. Cached per rope family per offset (`qkAngleFor`): 2 probe
   dispatches per token replace ~160 removed ones.
2. **Every rounding boundary spelled**: RMS mirrors `rms_single_row` at axis 128 (lane
   owns 4 contiguous elements, f32 squares in index order, one simd_sum,
   `precise::rsqrt`), the `T(x*inv)` rounding inside the w-multiply is the value the
   separate kernel writes, rotation is f32 with one T rounding, and mscale is a SEPARATE
   bf16 multiply after it (matching the shipped post-rope broadcast, 1.0 on the tail).
3. **The one bug found by the parity test**: the second-half rotation is
   `x1*sin + x2*cos` where x2 is the lane's OWN element — writing the naively-symmetric
   `partner*cos + own*sin` passes at offset 0 (identity rotation) and fails everywhere
   else. A host-side reconstruction from the probe angles (6e-8 max) localized it to the
   kernel in one step.

Measured: the eval-per-step fwd_ubench read the fusion +1.2% SLOWER; the LIVE paired A/B
(real generations, server timings, 2 pairs both orders) reads 8.94 -> 8.80 ms/tok, -1.6%,
tight spreads. Ship decisions come from the live number (the µbench forces a sync per
forward, which un-hides exactly the latency this fusion removes). Default ON for laguna;
`MLX_SERVE_QK_NORM_ROPE_FUSED=0` restores the composed chain. The qwen/gemma wiring
(gatedFullAttnWith / gemma4MoeAttnWith — every QK-norm arch has this chain) is the
documented next step; gemma's head_dim 256 needs a different lane mapping.

## Round 4 (2026-07-29): the remaining mlxfast levers — two ship, one is a µbench-vs-live scalp, one dissolves on contact with the checkpoints

Fourth mining pass over the mlxfast-challenge tree, executing the "next round" list from
`docs/perf-next-levers.md`. Outcomes, in the order attempted:

### Certified lm_head prune: kernels sound, argmax provable, LIVE NULL — ships opt-in

Port of `LagunaLmHeadPrune.swift` (notes/68): an init-time MXFP8-g32 copy of the dense
bf16 lm_head (half the bytes), one coarse GEMV emitting per-row coarse logit + a
CERTIFIED bound (half-ulp e4m3 cells + a 2^-15 rounding allowance, top cell 186) + a bf16
pre-fill; a dense one-byte candidate mask (`coarse+delta >= max(coarse-delta) - |L|/64`);
an exact pass whose per-row arithmetic textually replicates the stock
`gemv_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0` (verified to be what MLX dispatches
for this matvec) so every candidate row is bit-identical to the stock GEMV. Every
argmax-reachable row is provably a candidate; every non-candidate is provably below the
winner; the emitted token is the stock token.

Three lessons:

1. **The gate is REQUEST-level and must be known at init.** The pruned row is exact
   at candidates and coarse elsewhere — enough for an argmax, not for a tail
   distribution. `ForwardCtx.argmax_only` is set by the Generator (greedy or top-1, no
   penalties, no logprobs, no grammar), and logprobs had to move into `InitOptions`:
   the split-prefill final-token forward is a single-row dispatch that runs BEFORE any
   post-init `gen.logprobs_n = n` write, so a logprobs request would have had its first
   token's logprobs computed from a pruned row.
2. **A hermetic parity test cannot see reduction order at this scale.** A deliberate
   perturbation of the exact kernel's accumulation order (grouped instead of sequential)
   STILL passed the bit-equality test: an f32 last-ulp difference almost always vanishes
   in the final bf16 cast, and catching one empirically needs ~hundreds of thousands of
   casts. The live >=160-token greedy on/off run (16M casts) is the real bit-identity
   bar — same lesson as round 3's fused-QKV divergence at byte 164.
3. **The µbench-vs-live class, third instance, with the strongest live design yet.**
   µbench at the real [100352, 2048] geometry, with a separated-winner weight matrix
   reproducing the LIVE candidate regime (median ~19 of 100352 candidates, measured by
   `MLX_SERVE_LMHEAD_PRUNE_TRACE=1`): dense 1.04 -> pruned 0.63 ms/iter, a ~0.4 ms/token
   saving. Live: an INTERLEAVED same-boot A/B — the gate is per-request, and
   `presence_penalty: 1e-9` clears `argmax_only` while the lazy greedy sampler never
   applies penalties, so the dense arm is byte-identical work in the SAME boot,
   alternating per generation, and thermal drift cancels per pair. 20 rounds: median
   +0.84% per pair, 3/20 wins; the on-arm's early −1.8% advantage decays within a few
   generations while the off arm stays flat. The µbench saving does not survive the
   live graph, and the boot-level A/B that "showed" −1.3%/+1.6% was just drift. Where
   the ~0.5 ms goes live remains unattributed (candidates stay tiny all boot — traced;
   the threshold chain and select are flat in the stage bisection; the exact pass with
   a zero mask costs ~0.02 ms net). Shipped `MLX_SERVE_LMHEAD_PRUNE=1` opt-in with the
   trace + µbench left in place for whoever picks it up.

### MoE down-path tail fusion: −1.5%, bit-identical, default ON

`gatherQmvDownReduce`: the decode expert tail ran down-gather -> score multiply -> sum
over K as three serial dispatches with a [K, hidden] intermediate between them. One
kernel now does all three: each simdgroup owns a top-K slot and computes 4 output rows
(mlxfast tuned 1-vs-4 rows per simd; 4 won), parks bf16 row values in threadgroup
memory, and slot 0 finishes with the composed pair's exact semantics.

Bit-identity came from two replications, not one:
- the per-row dot is the SAME `GQMV_BODY_*` text `gatherQmv` compiles (same lane-strided
  pack loop, same 4-accumulator split, same `simd_sum((a0+a1)+(a2+a3))`, same 2^22 nvfp4
  fold, same `T(acc)` rounding), so per-(slot,row) values match by construction;
- the reduction replicates `mlx_multiply` + `mlx_sum_axis(-2)`: per-slot product rounded
  to T, then an ascending T accumulate — MLX's bf16 sum reduction accumulates in the
  INPUT dtype (the moeRouterTopK lesson generalizes from row_reduce to this col reduce).
  The parity test runs BOTH a small geometry and the real [1,1,8,2048] because MLX picks
  its reduce kernel by shape; both are max_diff 0.0 exactly, nvfp4 AND affine.

Declines are the kernel's own conditions, incl. scores dtype == activation dtype (a
mismatch would make the composed pair promote to f32 — replicating that would mean an
f32 tail, so it falls back instead). Live paired A/B on Laguna XS: 8.58 vs 8.71 and
8.56 vs 8.70 ms/tok, −1.45%/−1.53%, on-arm wins both pairs both orders. Greedy 200-token
on/off byte-identical. `MLX_SERVE_MOE_DOWN_REDUCE_FUSED=0`.

The reference's shared-expert + residual absorption (their 9th simdgroup) was NOT
ported: our shared-expert down runs through stock qmv, and absorbing it bit-identically
means replicating THAT kernel too; the residual+RMS+router fold
(`DARKBLOOM_FUSED_RESIDUAL_RMS_ROUTER`) was likewise deferred — two more stock-kernel
replicas for an upper bound of ~0.1–0.2 ms of mostly launch latency, in the same family
as two measured nulls (fusedAttnGate, fusedAddRmsNorm).

### QK-norm+RoPE for "qwen archs": the checkpoints say no

The port itself was small: rd=32 support in the shipped kernel (the qwen3.5/3.6
partial-rotary width; the `lane ^ (half_rd/4)` shuffle mapping is exact for 32/64/128),
wiring into `gatedFullAttnWith` after the `attn_output_gate` split (the strided q view
rides `ensure_row_contiguous` — parity-pinned including that exact case), M-RoPE decode
handled via the offset+delta the probe row reproduces exactly. All 4 parity tests are
max_diff 0.0.

Then the engagement check: SILENT. Every qwen3.5/3.6 checkpoint on disk — 2B, 9B, 27B,
35B — is **head_dim 256**, not 128. The "hd 128 fits as-is" line in the handoff doc was
written from the arch family's reputation, not the checkpoints. qwen sits in gemma4's
needs-a-new-lane-mapping bucket, and the only hd-128 QK-norm models here are the lagunas
(already fused). The wiring ships dormant (declines at hd != 128, verified byte-identical
composed-path behaviour on the 2B and the real 27B MoE+MTP stack), and will engage
unchanged on a future hd-128 arch.

### NVFP4-g16 tail for --decode-attn-quant: −3.3% more, quality holds, default ON

mlxfast quantizes attention layers >= 32 of 40 to real nvfp4 (late-layer quantization
amplification ~15x lower than early layers). Ours: `attnDqFor` threads the layer,
`attnDqUseNvfp4` puts the boundary at 80% of `num_hidden_layers`
(`MLX_SERVE_DECODE_ATTN_QUANT_NVFP4_FROM=<n>` moves it, `off` restores int8-only), and
`AttnDqCopy` carries its own (bits, group_size, mode) so the dispatch reads the copy's
geometry — a tail nvfp4 copy and a body int8 copy ride the same `qmatmulBits` call.
`buildAttnDqCopy` returns 2 parts for nvfp4 (no biases by construction; null-ctx handle,
deinit guards it).

Live paired A/B on Laguna XS (on top of the down+reduce fusion): 8.25 vs 8.52 and 8.26
vs 8.55 ms/tok, −3.15%/−3.39%, 2/2 pairs. The mandatory 6-prompt greedy
characterization (int8-only vs nvfp4-tail, temp 0, code + arithmetic + SQL + reasoning)
re-run: all six answers identical in substance (53361, 50 km/h, correct IPv4/SQL/
quicksort/binary-search), wording-level divergence only — the same class as the round-3
int8 characterization, so the tail rides the existing default-ON lossy toggle.

### Prefill sorted-MoE tail: already correct by construction

The investigation item ("does our sorted prefill path pay a scatterUnsort-style full
expert-bank copy?") closes with no change: our tail already does
`take_axis(down_squeezed, inv_order)` — an ACTIVATIONS-only gather through the inverse
permutation, which IS mlxfast's fix. The prefill fused gate+up bank (load-time weight
concat, additive memory) remains open in `docs/perf-next-levers.md`.

## A cached metal_kernel config keyed on element COUNT crashes the server on shape collision (2026-07-29)

**Live failure**: voice-mode chat on `poolside/Laguna-XS-2.1-NVFP4-mlx` killed the server after a
few turns, twice in one session. The log ends abruptly right after
`[hot-cache] reused 1015/1032 tokens (matched 1015)` — no error in the file (the MLX fatal goes to
stderr, which only the app's in-memory buffer sees), no crash report (the mlx-c error handler
exits). The app-side symptom read as "server crashes with a very long base64" because the debug
log's last big entries were the voice-clone TTS bodies (~550 KB of `ref_audio` each).

**Actual error** (recovered by replaying the session):

    MLX error: [matmul] Last dimension of first input with shape (1,16,512) must match
    second to last dimension of second input with shape (8192,2048).

**Root cause**: `fusedSwiGLU`'s cached `mlx_fast_metal_kernel_config` was keyed on
`{n, tg, ndim, dtype}` — element count, not shape — while the config bakes the builder's OUTPUT
SHAPE. On Laguna XS a MoE layer's SwiGLU at a 16-token forward has shape (1,16,512) = 8192
elements; a dense `mlp_only` layer at decode has (1,1,8192) = the same 8192 elements, rank 3,
bf16, tg 256. Equal key, different shape: the decode call was served the MoE config and returned
its activation labeled (1,16,512), and the dense down-proj (`x @ W(8192,2048)`) raised an
uncatchable frontend shape error → process death.

Why EXACTLY 16 tokens: 16 × 512 == 1 × 8192. And the only source of a 16-token forward is a
hot-cache-restore suffix prefill (cold prefills are whole prompts or ≥1024-token chunks; PLD
verify is ≤ depth+1). Voice mode was merely the environment that produced it: short spoken turns
→ the next prompt matches the stored entry to its FULL length → the unmatched suffix (turn
scaffolding + a short user sentence) lands near 17 tokens → 16 after t1 is split off. Live turns
died at suffix 17 and survived at 14/15/16/18 — the bisect that looked like a "≥16 kernel gate"
was really "suffix−1 == 16".

**Diagnosis path worth keeping**: the debug log dumps every request body whole, so the fatal
session is REPLAYABLE — extract bodies byte-exact (`[http] request body (Nb):` + N bytes) and
curl them back in order. A byte-exact serial replay did NOT reproduce (the fatal condition needs
the hot-cache entry to match the previous turn's own generated reply, and a replayed server
generates different tokens at temp 0.7); an INTERACTIVE replay (feed the server's own answers
forward) reproduced it on turn 2. Env-var bisect then took minutes: kv-quant off → still dies,
`--no-pld` → still dies, `--prefix-cache-entries 0` → survives, `MLX_SERVE_SWIGLU_FUSED=0` →
survives. Note the big TTS bodies are truncated at 16 KB in the log dump, so audio bodies can't
be replayed from the log alone (chat bodies can).

**Fix**: `ShapeKey` (full dims + rank) replaces every product-derived field in the four cached
config keys that bake input-derived shapes: `SwigluCfgKey`, `RouterCfgKey`, `AddNormCfgKey`,
`AttnGateCfgKey`. The gather-family keys (`GateUpCfgKey`, `GqmvCfgKey`, `DownRedCfgKey`) are
decode-only with fixed-rank shapes fully determined by {topk, n} and were left alone. Guard:
`test "fused SwiGLU config cache keys on the SHAPE, not the element count (Laguna restore crash)"`
— (1,16,512) then (1,1,8192), asserting the second output's shape.

**Class rule**: any future cached `metal_kernel` config whose config encodes an output shape,
grid, or template arg derived from an input shape must key on the FULL shape via `ShapeKey` —
"same element count" is not "same geometry". Also note the observability lesson: a mislabeled
elementwise output is byte-correct and only crashes when a downstream op checks shapes; a
BROADCASTABLE collision would have been silently wrong instead.

## Quantized-KV fused decode reads (2026-07-30, docs/kv-quant-perf.md round)

Three war stories from one round, each already distilled into a `## Rules` line:

**1. The fused flag was a silent no-op on the live decode path.** `--kv-attn-mode fused`
existed, parsed, was documented, had an equivalence script — and engaged on NOTHING. The
decode forward calls `cache.update()`, whose affine arm returned its `DenseKVView` WITHOUT
the quant triples; only the read-only `denseView()` (used by batched decode + diffusion,
both deliberately excluded) set them. Every "fused vs dense" measurement to date compared
dense against dense and read neutral. Same class as the two hardcoded `use_drafter=false`
call sites: output-equality tests structurally cannot see a silent fallback — the
equivalence script now FAILS unless the fused arm logs both `[kv-attn] fused engaged` and
`[kv-attn] decode kernel engaged` and the dense arm logs neither.

**2. `0 × -inf = NaN`, and every parity loop was NaN-blind.** The composed causal arm built
its additive mask as `triu(ones) * -inf` — below the diagonal that is 0 × -inf = NaN, which
softmax propagates everywhere. Live symptom: gemma-4-e4b answered `<pad><pad>` to any short
prompt under the (finally-engaged) fused mode; layer-by-layer live diff showed maxdiff=nan
from the FIRST "causal" layer onward. It shipped green because every parity comparison in
kv_quant.zig/transformer.zig used `if (e > max_err) max_err = e` — `NaN > x` is FALSE, so an
all-NaN candidate scores max_err 0 and PASSES. Both fixed together: masks via `mlx_where`,
and every parity loop asserts `!isNan` per element BEFORE the diff (red-on-revert verified:
un-fixing the mask fails 4 tests).

**3. The decode kernel's first two designs lost by 5x and 1.7x — both occupancy, not math.**
v1 (one threadgroup per q head, per-row simd_sum online-softmax chain) starved the GPU: 48
threadgroups, serial reduction+exp chain per row. v2 (block-parallel two-phase, 2048-row
blocks) had the right shape but 24-28 KiB of threadgroup memory per group → ~1 threadgroup
resident per core → no latency hiding (1.9 ms vs dense 1.1 at 48/8 32K). Same kernel at
≤10 KiB (256-row blocks) wins every ≥32K cell. The µbench→live ladder: 5.7 → 1.9 → 0.83 ms
against dense 1.10. Live A/B (same boot, per-request `kv_attn_mode` interleave, prefix cache
serving the long prompt so requests measure DECODE): Laguna XS +10% @10.7K, +56% @42K;
Qwen3.6-27B +26% @37K — and a −2% regression when Laguna's 512-window sliding layers fused,
which is where the per-layer `KV_ATTN_FUSED_MIN_TK` floor comes from. The wired-site lesson
repeated mid-round: the first Laguna A/B read EXACTLY neutral because `lagunaAttnWith` was
not yet wired and both arms ran dense — the engagement grep (0) was the tell.

## The SSM-checkpoint stride was quietly the biggest prefill lever in the engine (2026-07-30)

llm_context_benchmarks read mlx-serve 19-20% behind oMLX on Qwen3.6-27B prompt processing
at 8K/32K (196.9 vs 243.6, 179.6 vs 224.6 tok/s) while we WON decode — and the falloff
shape (us 230→197→180 across 2k/8k/32k, them flat) pointed at something that engages
between 2K and 8K. `MLX_SERVE_PREFILL_TRACE=1` showed an 8238-token prompt prefilling in
**33 chunks**: the dense-hybrid arm of `effectiveSsmCheckpointStride` kept the raw
256-token stride, and `nextChunkEnd` dutifully ended a chunk at every 256-boundary. The
old rationale ("dense prefill is compute-bound, extra chunks are ~free") pre-dated
`prefillDqGemm`: with the dq route gated at M ≥ 2048, EVERY projection of EVERY chunk ran
the slow small-M qmm path, plus 33 graph builds, eval barriers and per-chunk MTP history
captures. One flag flip (`--ssm-checkpoint-stride 2048`) recovered +19% at 8K same-session.
Fixes, each measured paired on the 27B (M4 Max):

1. **Stride never sub-divides the prefill chunk, on ANY arch** — the MoE coarsening arm
   is now universal (`effectiveSsmCheckpointStride(base, prefill_chunk)`); tiny tails also
   merge under checkpoint alignment (a 1-token trailing chunk existed only to lay a
   snapshot one token before the always-on end snapshot).
2. **SSM states are materialized in EVERY chunk's eval batch** (they were gated on
   stride-aligned boundaries): a tail-merged final chunk ends off-boundary, and the
   always-on end snapshot then captured an UN-evaluated state — `materializedOwnedCopy`
   re-executed the whole last chunk's GDN scan (−4% at 8K, invisible in the trace's
   `eval=` bucket because the re-execution bills to the capture).
3. **Dense hd-256 fused-causal chunk cap is 8192; MoE keeps 4096** (`boundedPrefillChunk`
   grew an `is_moe` param): dense hybrids have no expert-gather transients, and the full
   chunk halves per-chunk dequant sweeps (+1.4% at 8K, flat at 32K). The gemma-26B@99K
   "+3% for +22 GB peak" lesson still governs MoE.
4. **The always-on snapshot sits `SSM_SNAPSHOT_BACKOFF` (30) tokens BEFORE prompt end,
   and the held-back tail rides the final logits forward.** A snapshot exactly at the
   prompt end is unreachable for the next turn's prefix match: the template's
   generation-prompt suffix (`<|im_start|>assistant\n` + think opener) renders differently
   once the turn enters history, so the match always lands a few tokens short —
   `[hot-cache] hybrid miss (no checkpoint ≤ 870 of 897)` was llmprobe's
   prompt-cache-prefix cell failing. This is the REAL mechanism behind the 2026-06-10
   "coarsening disabled every prefix-cache hit" event; the fine stride never fixed it, it
   just happened to lay boundaries underneath the match point. Backoff sizing matters
   twice: 64 pushed the tail forward to seq ≥ 32, which `prefillEvalCadenceApplies`
   treats as a prefill — ~450ms of mid-loop eval bubbles on the 27B; 30 keeps it on the
   verify-shaped fast path (~1ms). MTP history for the held-back span is appended from
   the final forward's capture-all (a history hole right before the generation point is
   acceptance-critical).

Result (same-session, defaults vs defaults): 2k 268 vs oMLX 243 (+10%), 8k ~tie at 252,
32k ~tie at 224 — from −4/−19/−20%. The 8K/32K ties are physics, not a remaining bug:
2×27e9 FLOP/token at MLX's measured 14.4 TFLOPS bf16 GEMM rate (89% of the M4 Max's
theoretical 16.2) puts the 8K roofline at ~30.9s and both engines within 3-6% of it.
Nobody beats anybody by 5% on a dense-27B prefill on this hardware; the winnable margins
live at short contexts (fixed overheads) and on MoE/small models.

### A synthetic-dtype reference probe nearly shipped a 2x-bandwidth Inkling forward (2026-07-30)
Porting Inkling Small, the dtype question was "does the residual stream run bf16 or f32?" — the reference multiplies every dense-MLP output by a `[1]` `global_scale` tensor, and an early python probe (reference modules, MY casts: global_scale → f32 like the "keep_hi" converter comment implied) showed bf16 × f32-array promoting the whole stream to f32 from layer 0. Plan accordingly: f32 KV, f32 experts, 2x bandwidth. WRONG: the REAP25 checkpoint STORES the dense `mlp.global_scale` tensors as BF16 (the base model's were bf16, so the converter's f32-keep condition never fired); only the ROUTER's `gate.bias`/`gate.global_scale` are f32. The real stream is bf16 end-to-end. The probe proved the reference's promotion SEMANTICS while saying nothing about the checkpoint — same family as "read the CHECKPOINT, not the reference source" (Kokoro AdaIN, laguna YaRN), one level up: read the checkpoint's DTYPES, not the converter's intent.

What caught it: the mandatory `[dtype-trace]` arm on first live boot read "residual widened at layer 2: bfloat16 -> float32" — layer 2, not 0. Layers 0-1 (dense) NOT widening disproved the f32-global_scale theory on the spot, and the layer-2 widening localized MY deviation: the routing weights came out of the f32 router chain and multiplied the bf16 expert outputs un-cast, where the reference does `topk_weights.astype(x.dtype)` (and rounds the shared gammas through x.dtype before its f32 shared sum). Fixed to mirror the reference; greedy output then matched the ground truth byte-for-byte on the 32-token prompt (the second prompt diverged at token ~12 into an equivalent continuation — the sanctioned INT4 kernel-order class). Related trap in the same round: the load-time fold of `gate.global_scale` into `route_scale` reads the tensor via `mlx_array_data_float32`, which returns null on a non-f32 tensor — the fold would have been a SILENT no-op had that tensor been bf16 too. Dtype-gate any raw-data read whose tensor dtype you didn't verify.

## mlx_array_new_data copies shape-worth of bytes: the guard-page latent test bug (2026-07-31, dsv4 port session 2)

`transformer.zig`'s "fused residual+RMSNorm declines what it cannot reproduce" test needed a
[1, 8192] array only for its SHAPE (the fusion's decline path never reads the data), so it
passed a 16-float stack buffer with an 8192-wide shape. `mlx_array_new_data` memcpy's
shape×itemsize bytes — a 32 KB read out of a 64-byte buffer. That's UB, but stacks are
mapped generously and the test stayed green for weeks; the dsv4 GPU-chain edits shifted
code (and thus frame layout) elsewhere in the binary, the buffer landed near the stack
guard page, and the memcpy Bus-errored — in a test whose subject had nothing to do with
the change. Fix: the buffer is really 8192 floats now. Rule: every `mlx_array_new_data`
buffer must be at least shape-product elements, even when "only the shape matters".

## dsv4: module-owned decode state vs prefix cache + Transformer teardown (2026-07-31)

DeepSeek-V4-Flash native keeps ALL per-request state (raw-kv GPU rows, compressed caches,
compressor pending rings) on `Dsv4Model.dec_state` — the Transformer's KVCache is a
0-entry shell whose only live field is `cache.step`, and the serving seam keys on
`step == 0` to rebuild the state. Two lifecycle consequences, both found by review before
they shipped:

1. `HotPrefixCache.shouldUse` returned true for dsv4 (no hybrid layers), so a prefix cache
   would attach, match a repeated prompt, trim it and set `cache.step != 0` — WITHOUT
   restoring the module-owned state. Best case that silently serves the previous
   request's rings (often "works" on bench re-runs of the same prompt — the worst kind of
   wrong); on a fresh boot `mdl.dec_state.?` is null → unreachable = ReleaseFast UB.
   `shouldUse` now rejects `deepseek_v4` by model_type until the dsv4 state rides the
   ssm-entry machinery. Guard: prefix_cache + scheduler unit tests.

2. `Transformer.deinit` had no dsv4 arm — the BERT shell inlines its fields into the
   Transformer (freed by the normal paths), so the MODULE-POINTER pattern (`allocator.create`
   in initDsv4) had no teardown precedent, and unloading a dsv4 model leaked the entire
   ~108 GB of mlx handles + host arenas. Any future module-owned arch (the dsv4 pattern is
   the template) owes the same deinit arm.

## dsv4: the PLD guard bypass — a guard that shapes init options does not bind dispatch (2026-07-31)

The live tool-call request on the DSV4 mirror came back with mangled DSML (dropped token
runs, wrong-order tag fragments leaked into content) despite TWO spec guards reading as
airtight: `scheduler.runPrefill` computed
`is_dsv4 = slot.model.transformer.?.dsv4 != null` and hard-off'd `use_pld`, and (added the
same session) `Generator.initWithOptions` chokepoint-forced `options.pld_enabled=false`
when `xfm.dsv4 != null`.

**The log evidence and the illusion.** `~/.mlx-serve/logs/mlx-serve-11234.log` 166348–166361:
on the two tool-less requests, `spec-gate: ngram-score=0.000` → `pld=disabled (ngram…)` — but
those lines are printed by the CONN THREAD in server.zig, before the scheduler ever sees the
request. They prove nothing about what the generator ran; "the guards held" was the ngram
gate declining naturally. On the tool request the tools JSON is repetitive, the ngram gate
passed (0.093), and the smoking guns were generator-internal lines:
`pld=disabled (yield gate: 0 drafted tokens over 8 steps)` (generate.zig) and
`[spec-stats] mode=pld attempts=2` — `pld_attempted` increments only after a VERIFY forward.

**Root cause: the decode tick dispatched on the slot flag alone.** `runSingleDecodeTick`:

- `if (slot.enable_mtp and gen.mtp != null)` — generator-state conjunct, chokepoint nulls `gen.mtp` → safe
- `if (slot.enable_drafter and gen.drafter != null)` — same → safe
- `if (slot.enable_pld)` — NO generator-state conjunct (PLD has no model handle, so there
  was nothing natural to check) → every tick called `gen.nextPld` regardless

And `nextPld` trusts its caller: it checked only `done` / `specDecodeUnsupported` /
`spec_disabled_runtime` — `InitOptions.pld_enabled` only ever influenced init behavior (skip
of the lazy preforward, a log suffix). So both guards worked exactly as written and neither
mattered: the tick ran nextPld's lookup, found matches in the repetitive tool schema, ran
verify forwards `[t1, draft…]` through `forwardWith` → dsv4's module-owned decode state
(compressor rings, kv/comp GPU caches, position bookkeeping) appended 1+m tokens, and the
KV-snapshot rollback restored only the KVCache SHELL. Two rejected drafts left the module
state permanently ahead of `cache.step` → every later token attended at wrong
positions/windows → mangled DSML. mtp/drafter were saved by an ACCIDENT of representation,
not by design.

**Fix (both sides must agree, and the generator defends itself):**

1. `Generator.pld_enabled` field, set from the POST-chokepoint options — PLD's explicit
   counterpart of `gen.mtp != null`.
2. `nextPld` self-declines at the top when `!self.pld_enabled`: delegates to the plain
   serial `next()` (single-token PldStepResult). Unlike `spec_disabled_runtime` this is
   permanent — the mid-request re-enable check can never resurrect it (the generated tail
   of a tool turn IS echo-heavy; a resurrectable disable would re-corrupt).
3. Tick dispatch through the pure `scheduler.specTickMode(slot flags × generator state)`,
   hermetic contract test: slot-wants-pld + generator-not-armed → `.regular` for all three
   modes, plus the MTP > drafter > PLD priority.

**Regression test trick** (`generate.zig` "nextPld on a chokepoint-disabled generator stays
serial", DSV4_MINI-gated): a random-content prompt can idle in PLD's cold path forever and
mask the bug — the corruption needs a LOOKUP MATCH. Prompt = every vocab id once (mini
V=64) + `key_len=1` makes any sampled t1 match an earlier position, guaranteeing a draft
and therefore a verify forward on the broken code (red read `pld_attempted=5`; green
requires 0 AND byte-identical tokens vs a serial-arm run on fresh weights).

Audit of the other non-Generator paths: ds4/llama/diffusion slots route out of
`runSingleDecodeTick` before the spec arms (session pointers / runner checked first) — not
exposed to this class. Encoder-only models never reach a generator (`textGenRejectReason`).

Bonus find while running the gates: the DSV4_MINI fixtures parity test leaked its
`loadDsv4Weights` layer table on the skip path (`readAll(fixtures.json) catch return` sat
AFTER the weight load; ownership only transfers at `initModel`). Skip-path reads now come
first.

## dsv4 DSpark: lazy stage weights + Metal's zero-filled OOM = fake 100% acceptance of `<BOS>` (2026-07-31)

**Symptom** (real mirror, `MLX_SERVE_DSV4_DSPARK` default-on): greedy chat answered ~2
correct tokens then `<｜begin▁of▁sentence｜>` (token 0) forever; `[spec-stats]` read
`accepts=15 avg_per_round=5.00` — a fake 100%. `=0` arm byte-correct. Three mysteries had
to explain together: (a) stage-2 `moeGpu` returned EXACT zero with healthy inputs AND
healthy weights (ones-probes read real norms); (b) round-1 VERIFY rows — the trunk path,
pinned 6/6 on the mini — all argmaxed 0 on a clean prefill; (c) draft `conf` values
differed per BOOT for the identical request (1+hc_eps / the fp8 amax floor / 0 — "stale
pool bytes").

**Root cause A (the production bug)**: the DSpark stage weights were never materialized.
MLX loads are lazy; the house materializer is the WARMUP forward — and dsv4 is the first
arch whose warmup does not touch every loaded weight (stages only run in the draft). The
first `dsparkRound` first-touch-materialized ~10.9 GB of 4-bit expert banks MID-REQUEST,
on top of a 110.8 GB resident trunk, into a 118 GB `max_recommended_working_set_size` box.
The `[dspark-trace]` active/cache/peak counters caught it as a clean ramp: 110.8 GB →
+1.15 GB per expert bank touched → 117.9 GB at stage-2's w2 gather — where Metal command
buffers began failing `Insufficient Memory`. **Failing command buffers hand back unwritten
(zero-filled) buffers with NO surfaced error for many evals** — draft logits all zero →
argmax token 0; verify logits all zero → argmax token 0; they "agree", so the accept loop
committed 5/5 `<BOS>` per round. The uncatchable MLX abort only fired boots later (or not
at all), and where the cliff landed in the ramp varied with pool state — mystery (c)'s
boot-to-boot nondeterminism. Every earlier structural suspect (ShapeKey collisions,
arena/lifetime bugs in the draft) was innocent; the all-zero + independent-chains +
nondeterminism triple is a MEMORY signature.

**Fix**: `initModel` collects every stage tensor via a comptime-reflective walker
(`appendWeightArrays` — Q triples, optionals, nested structs; a future field cannot be
silently left out), then decides `dsparkFitsBudget(stage_bytes, trunk_bytes, max_rec,
6 GB headroom)` on LOGICAL bytes (size × itemsize, known pre-eval) — at init the TRUNK is
also still lazy, so `mlx_get_active_memory` sees neither side and cannot feed the
decision (the first fix attempt used active and admitted an 11 GB overflow). Fits → one
batched `mlx_eval` pays the true footprint at load, where preflight/wired/auto-context
can see it. Doesn't fit → `n_mtp = 0`: DSpark disabled with a log naming every number,
serial serve, and the untouched lazy stages cost zero bytes. `MLX_SERVE_DSV4_DSPARK=1`
(explicit) forces past the check for paging experiments. On the 128 GB box the guard
fires (trunk 101.4 GB logical + stages 11.0 + headroom 6.1 > 118) — the 4-bit-stage
mirror structurally cannot serve DSpark there; that is now an honest boot-time line, not
mid-request corruption.

**Root cause B (found by the same session, SIGBUS)**: `toHostF32` did astype→eval→
`mlx_array_data_float32`→memcpy. `mlx_astype` to the SAME dtype is a no-op VIEW, so a
strided/broadcast **f32** input (the dsv4 stream is f32 by design) kept its strides and
its raw buffer held FEWER elements than the logical count — the memcpy overread stale
pool bytes (garbage norms in the trace; conf values echoing unrelated constants) and, one
boot, crossed into another thread's stack guard region (SIGBUS in `_platform_memmove`,
"crash was associated with thread 3 — possible stray access"). Fix: check
`isRowMajorContiguous` AFTER eval (strides are unpopulated before), materialize via FLAT
RESHAPE — a non-row-major layout can never be expressed as a 1-D view, so MLX must copy.
Add-scalar-zero (the `materializedOwnedCopy` pattern) does NOT materialize a broadcast:
MLX binary ops propagate the broadcast input's strides to the OUTPUT and compute only the
unique elements (measured: post-add strides still `{0,1}`). bf16 inputs were always safe
(cross-dtype astype materializes) — the class needed an f32-stream arch to surface.

**Class guards**: "toHostF32 reads non-contiguous f32 views in logical order" (broadcast
+ column-slice); `dsparkFitsBudget` unit test with the real mirror's numbers; collector
assertions in the DSV4_MINI load test; and the FULL-ACCEPT seam test
(`dsparkRoundWith` + a draft rigged to the serial continuation) covering the no-rollback
branch the random mini can never reach — commits the block, bonus token correct, 3 serial
tail tokens bit-identical after the round.

## DSpark round-cost round: the barrier, not the transfer (2026-07-31, dsv4)

DSpark shipped correct but SLOW: 13.5–14.0 tok/s against 22.6 serial on the same box.
The plan's first item was a per-round cost audit, and doing that before touching anything
is what kept three plausible theories from being implemented in the wrong order.

**The profiler is honest only because the phases already synced.** `MLX_SERVE_DSPARK_PROFILE`
wall-clocks draft / snapshot / verify / rollback. Unlike `MLX_SERVE_DECODE_PROFILE` (whose
per-phase evals kill pipelining and make it a lying sizing tool) every boundary measured
here is ALREADY a host sync in the shipping path: the draft ends in the confidence read,
both verify paths end in a logits read, the snapshot is pure host work. Zero added evals.

First reading, 195 ms/round committing 4.69 tokens (43 ms/token): **verify 142, rollback 40,
draft 12.6 (markov 5.0), snapshot 0.24.**

**Trap inside the audit: a phase that is zero on some rounds must not be read as a per-round
average.** Rollback showed 40 ms/round, which looked like 20% — but it is 0 on every full
accept, so the real partial-round cost was ~65 ms. The average was right; the interpretation
("this is a fifth of the round") was not, and it nearly demoted the biggest lever.

### 1. A rollback that re-forwards the accepted prefix is a SECOND forward (+34%)

The partial-accept path restored the entry snapshot and re-ran `extendState` over
`accepted+1` tokens — a whole batched trunk forward to reproduce state the verify had
already computed. The comment justifying it was true but too pessimistic: "the compressor
pending rings are overwritten in place, so partial-position rollback has no anchor short of
the snapshot". Everything ELSE already rolls back by offset (GpuRows `used`, the append-only
compressed caches, `st.n`, the DSpark main_kv rings). So capture ONLY the rings, per token,
while the verify runs: ~12 MB of memcpy for a whole block, measured at **0.24 ms**, replacing
a ~65 ms forward. Round 195 → 156 ms, decode 22.9 → 30.7 tok/s.

`DsparkAnchors` captures inside the compressor push loop (`captureComp`), so position p holds
the state after p+1 tokens; `restoreToAnchor` takes offsets from the entry snapshot and ring
contents from the anchor. Guard: "anchored rollback replays like snapshot + re-extend" runs
BOTH strategies from the same entry state at every acceptance count and demands bit-identical
tail decodes — proven red by sabotaging the `sc_pend` copy.

### 2. A host read inside a layer loop is a GPU BARRIER (+16%)

With the rollback gone, verify was 142 of a 156 ms round — and sub-lapping it showed the
vocab head was only 5.5 ms (the M=B+1 qmm cliff was the obvious suspect and was innocent).
The real number: **the per-layer compressor-input read was 128 ms of the 143 ms verify.**

The read itself is a few KB. What costs is that each of the 41 reads DRAINS the queue and
then idles the GPU across the host push loop, so a 43-layer forward is chopped into 41
serialized stages. The tell was a comparison the profile made free: a batched forward at
C≈1.6 already cost ~70 ms, while a serial `decodeStep` at C=1 costs 44 — the batched path
was paying a fixed tax per layer that decode had already fixed for itself (perf round 10's
`processDeferredComp`, +21.8% at the time, deferred exactly these reads).

Generalized to chunks: a chunk owes the read only if it CLOSES a compression window, because
only then is a slot emitted that the same chunk's later tokens can see. That is pure position
arithmetic (`chunkCrossesBoundary` — no data dependency), so the decision is exact rather
than heuristic. At a C=6 verify the 20 ratio-128 layers defer into ONE batched eval after the
layer loop; the 21 ratio-4 layers always close a window and still stall. Round 156 → 136 ms,
decode 30.7 → **35.3 tok/s**.

**The remaining lever is now measured, not guessed**: sync is still 109 ms of the 124 ms
verify, all of it the ratio-4 layers, and the only way to remove it is to compute the
compressor emission itself on the GPU (pending ring mirrored GPU-side, masked softmax combine
+ norm/rope/QAT-sim, ring update by slice_update). Estimated ceiling if the barriers go: verify
~50-60 ms, round ~70 ms, ~15 ms/token. That work also lands on batched prefill and on decode's
remaining boundary syncs.

### 3. Two levers that measured NEGATIVE, and why they stay in the tree

**The confidence gate.** antirez's ds4 makes the checkpoint's confidence head its biggest
lever: it truncates the submitted block (and skips the draft LM head entirely below the
threshold), default `--dspark-confidence 0.9`. Our head is equally informative — harvested
over 86 live rounds, position 0 scores mean **+5.08** on rounds that accept it vs **+0.02**
on rounds that reject it. It still lost: no gate 30.7 tok/s, 0.5 → 29.9, **0.9 → 25.3**.
The reason is in the same profile: our batched forward is FIXED-COST dominated. Fitting
verify(C) across three live arms (C=6 → 143 ms, 2.86 → 109.5, 1.61 → 87.6) gives roughly
**70 ms + 12 ms·C** — a narrower block pays nearly the same and commits less. The gate is
shipped OFF behind `MLX_SERVE_DSV4_DSPARK_CONF` (sigmoid units, matching the reference's
CLI) with the numbers in the source. A lever's default belongs to the engine that measured
it, not to the engine it was ported from.

**The sinkhorn config cache.** `hcPreBatch` rebuilt an `mlx_fast_metal_kernel_config` per
call — 86 per batched forward — which is textbook "a config rebuilt per call is a CPU-side
tax that reads as kernel cost". Caching it by token count (`sinkhornCfgFor`, the house
ShapeKey discipline: cache small repeating widths + the prefill sub-chunk, never one-off
remainders) measured as a wash. Kept because it is strictly less work and the table is
bounded, but it is NOT where the fixed cost was — the barrier was.

**Also fixed in passing**: `extendState` cleared the MLX allocator cache after every
sub-chunk, correct for prefill (shapes never repeat across prompt lengths) and exactly
wrong for spec widths (≤ block+1, repeating every round) — each clear made the next verify
re-allocate its transients from the OS. `extendChunkShouldClearCache` keys on the house's
"a multi-token forward is not a prefill" line (seq ≥ 32). Worth ~3%.

### Result

Same box, paired boots, engagement proven per arm (`[spec-stats] mode=dspark`,
`sub/round=5.00`): serial **22.6/22.7**, DSpark **35.2/35.3 tok/s** = **1.56x** serial and
1.32x the ds4 GGUF engine's 26.68. Prefill unchanged (135.9 / 143.2 / 127.8 tok/s at
488 / 1970 / 7823 tokens). Quality gates: `test_dsv4.sh` 14/14 + template A/B 17/17; the
5-task set 5/5 native (serial AND dspark) vs 5/5 ds4.

**DSpark ON is not token-identical to serial** — the documented near-tie kernel-choice class
(batched [C] verify vs [1] decode) — and the task set makes that concrete: serial answers
"17 * 23?" with a bare `391`, DSpark with a correct but rambling paragraph. Also honest:
on the code task BOTH our arms answer `sum(range(1, 11))**2` (wrong — square of the sum)
where the ds4 GGUF answers `sum(i*i for i in range(1, 11))`. That is identical in serial, so
it is our quant mix (2-bit gate/up, 3-bit down, no imatrix) against antirez's imatrix-
calibrated IQ2XXS/Q2_K, not the engine and not DSpark.

## The prefill admission guard billed a dense attention shape for a sparse arch (dsv4, 2026-07-31)

A 5806-token prompt to DeepSeek-V4-Flash — a 1M-context model — came back:

```
400: Prompt (5806 tokens) requires ~10277MB GPU memory but only ~8629MB available.
```

10 GB of working set for 5.8K tokens is absurd on its face, and the number reproduces exactly from `server.prefillMemoryNeeded` at that checkpoint's config. Two independent terms were wrong, and they happened to be the two biggest:

| term | billed | truth |
|---|---|---|
| scores | 3992 MB | ~440 MB |
| 3×mlp | 3960 MB | ~1050 MB |
| kv | 259 MB | fine |
| dequant | 11 MB | fine |

**1. The score term assumed dense causal attention.** `head_dim: 512` is outside every fused kernel (`prefillHeadDimFused` covers ≤128 and 256), so the composed-SDPA score scratch is billed — as `heads × chunk × seq`. But DSV4 is sparse: `deepseek_v4.zig` builds `tk = wk + n_sel` with `wk = @min(m.window, seq_total)`, i.e. a 128-wide raw sliding window plus ONE compressed arm — top-`index_topk` (512) of the `seq/4` slots on ratio-4 layers, or all `seq/ratio` slots visibility-masked otherwise. So a query reads **at most 641 keys at this length, not 5806**, and the over-bill grows linearly with the prompt while the truth stays nearly flat. Note the all-visible arm IS seq-scaled (`seq/128`), so it overtakes the top-k arm at long context — the bound tracks the widest layer rather than freezing at `index_topk`.

**2. The FFN width was a struct default the checkpoint never stated.** DSV4's config ships no `intermediate_size` at all, so `ModelConfig`'s 15360 — a Gemma shape — leaked into `@max(intermediate_size, moe + shared)`. The code even carried a comment acknowledging this ("a fat ffn costs MBs of estimate, not GBs"), which holds only for small chunks; at a 5632-wide forward it cost 3.96 GB. `intermediate_size_declared` now distinguishes "the JSON said 15360" from "nobody said anything".

The first draft of that fix billed the bare `moe_intermediate_size` (2048) and was **wrong in the dangerous direction**: a MoE chunk gathers `num_experts_per_tok` expert rows per token, so the transient scales with `top_k × moe_intermediate + shared` — 12288 here, not 2048. Under-billing this guard does not produce a 400, it produces an uncatchable Metal OOM that kills the process, so the asymmetry decides every judgement call in it.

Corrected: **4848 MB**, admitted with room. A sweep over every checkpoint on disk moved exactly two bills and left the rest byte-identical: DSV4 (10277 → 4848 at 5.8K, 9385 → 4225 at 64K) and Qwen3.6-35B-A3B (3915 → 1395 at 5.8K), the latter purely from the width fix — its real per-token MLP is 8 experts × 512 + 512 shared = 4608, against the 15360 it was being charged.

One honest caveat: the corrected number is not the true peak either. DSV4's indexer builds a `[chunk, heads, S]` similarity over compressed slots (`S ≈ seq/ratio`) that the estimator does not model at all. The formula was simultaneously over-billing two terms and missing a third; it just happened to land net-conservative.

Guards: `prefillAttnKeys` (model.zig — dense archs, the ratio-4/all-visible crossover at 1M, short-prompt clamp, and the no-ratios dense fallback), `prefillFfnWidth` (declared-beats-default, shared adds, dense unaffected), the end-to-end KEY BOUND estimate pinning both the 10277 MB failure and the 4848 MB fix, and a source scan asserting `checkAttentionMemory` passes `config.prefillAttnKeys(seq)` rather than `seq` — the engagement class, since handing the new parameter `seq` reproduces the old bill exactly while every direct unit test still passes.

## DSV4 serial-perf round: the sorted-gather 2x and the wrong barrier theory (2026-08-01)

Goal: push DeepSeek-V4-Flash serial decode + prefill toward the hardware limit, no DSpark, no quant changes. Result: decode 22.86 → ~26.7 tok/s (+17%), 8K prefill ~133 → ~267 tok/s (2x), greedy output byte-identical, test_dsv4.sh 14/14 on the final tree.

**The wrong theory.** TODO carried a measured claim: the host compressor emission's in-layer `toHostF32` barrier (41 per chunk) was the dominant prefill cost (109 of 124 ms in the DSpark verify at C≈6). A per-token/per-chunk phase trace (`MLX_SERVE_DSV4_TRACE=1`) seemed to confirm it: `comp` (sync + host pushes) was ~3.8 s of a ~3.8 s chunk. GPU window emission (`emitWindowsGpu` — masked-softmax combine, norm, rope, QAT sims, all in-graph, ring rows uploaded from the HOST rings which are bit-identical by construction) removed every barrier — and the chunk time did not move: the cost migrated wholesale into the one deferred eval. The barrier was never the cost at prefill scale; the cost was GPU compute — specifically the **unsorted expert-bank re-read**: `moeGpu` called `gather_qmm` with per-token indices and `sorted_indices=false`, so a 512-token chunk streamed each layer's ~1.9 GB expert bank ~12x. Porting the `_gather_sort` pattern (already in `moeMLP2` and `inklingExpertsApply`) halved the chunk: 3.7 → 1.85 s. Lesson restated: the C≈6 verify measurement was real, but extrapolating its attribution to C=512 was not — barrier cost per chunk is ~fixed while compute scales with C.

**Host cache content is dead on GPU streams.** With GPU emission authoritative for the mirrors, the full host `compressorPush` (f64 softmax combine + norm/rope/QAT sims, ~millions of scalar exps per chunk) survives only for ring maintenance and cache LENGTHS: mirror appends are suppressed, `DsparkAnchors`/snapshots compare lengths, restore only truncates. `compressorPushLight` = ring memcpys + fused ape add + `appendNTimes(0, d)`. The CPU stream keeps the full host path (the python-oracle and strict decode-equivalence gates model it).

**Decode wins, in order.** (1) wo_a served quantized: the [og, gin, ol] bf16 `wo_a_deq` slabs (67 MB/layer = 2.9 GB/token, the single largest read) replaced by reshaped views + batched `quantized_matmul` — +6.3%, prefill-neutral, and the deq cache and its load-time dequant are gone. (2) Deferred comp rows async-scheduled with the head walk (they are side branches the logits cone never computes; the old separate mini-eval was 2.75 ms/token). (3) hc-head sigmoid mix on GPU (the last mid-head host sync). (4) Lazy pipelined decode: GPU bf16 embed table (~1 GB, the RAM wo_a_deq freed) + device tid2eid hash lookup + lazy [1, vocab] logits let dsv4 ride generate.zig's pipelined next(); pending ring pushes drain at the next window-boundary token (`drainPending`), +6%. (5) `dsv4_sinkhorn_y` (sinkhorn + y-collapse, one dispatch, threadgroup-shared pre) and `dsv4_hc_post` (combT@stream + post·out from the raw pack) cut the hc chain from ~14 to ~6 ops per sublayer, +2%.

**Attribution probes** (timing-only, garbage output, live in `decodeLayers`): `MLX_SERVE_DSV4_LAYER_CAP=N` and `MLX_SERVE_DSV4_SKIP_MOE=1`. Measured: 0.91 ms/layer, ~zero fixed cost, 61% attention / 39% MoE at decode; prefill after the sort is ~ALL attention-side (skip-MoE prefill ≈ full). `MLX_SERVE_DSV4_PREFILL_SUB=1024` was a wash — prefill is not per-expert-M-bound.

Kill switches: `MLX_SERVE_DSV4_WO_QMM`, `MLX_SERVE_DSV4_GPU_EMIT`, `MLX_SERVE_DSV4_LAZY_DECODE`, `MLX_SERVE_DSV4_SINKY`, `MLX_SERVE_DSV4_HCPOST` (=0 each). Guards: DSV4_MINI suite incl. the new "lazy pipelined decode matches decodeStep" and "wo_a batched qmm slabs" tests; fallback arms re-run with each switch off.

## DSV4 AR kernel round: the decode-chain 8.6% and two measured-negative fusions (2026-08-01)

Follow-up to the serial-perf round, working the lever list top-down (emission kernel → RMS+rope chains → MoE gate+up A/B → drain prefetch → sink-softmax). Result: serial decode ~26.9 → **~29.6 tok/s (+10%)**, all DSV4_MINI gates + test_dsv4.sh 14/14 green on the final tree, DSpark healthy (31-48 tok/s content-dependent, acceptance 1.8-3.4/round).

**Shipped default-ON (kill-switched):**

- `dsv4_emit_win` (`MLX_SERVE_DSV4_EMIT_KERNEL=0`): one threadgroup per closed window runs the whole boundary emission (ext-row indexing over pre-ring/comp_in + ape, masked-softmax combine, RMSNorm, rope tail, Hadamard+fp4 or fp8 sim) that was ~60 composed ops per compressor. Serves decode AND chunk/verify paths so decode-vs-reforward equivalence compares kernel to kernel. **+1.4-1.7%**.
- `dsv4_dec_chain` (`MLX_SERVE_DSV4_DEC_CHAIN=0`): per-head [RMS →] rope-tail → [fp8-head-sim | Hadamard+fp4] in one dispatch for the four decode chains (q, kv-write, indexer-q, o-inverse) — ~80 strictly-serial glue ops/layer → 4 dispatches. **+8.6%** (27.2 → 29.6), the round's big lever. The attention-side "35 small ops around three qmvs" chain latency was real.
- Boundary drain prefetch (`MLX_SERVE_DSV4_DRAIN_PREFETCH=0`): the step BEFORE a window-closing token drains the OLDER pending compressor rows (their async evals finished a token+ ago), so the boundary token's blocking drain shrinks to the latest token's rows. Bit-identical by construction; measured a WASH on throughput — kept for the flatter boundary-token latency, and because it is free.

**Shipped OPT-IN (measured against, kept for future geometries):**

- `dsv4_moe_gateup` (`MLX_SERVE_DSV4_MOE_GATEUP=1`): the transformer.zig gatherQmvGateUp pattern with dsv4's clipped SwiGLU (f32 chain, sigmoid from an exact bf16-indexed mlx_sigmoid table). On the real 2-bit gs64 banks it LOSES ~2.5% to the two stock gather_qmm dispatches (29.6 → 28.9) — the "stock gather wins where O(bank) doesn't dominate the expert math" outcome holds even at MLX's slowest bit-width, k=6 over 256 experts just doesn't leave enough win per launch. Notably the fused kernel is MORE accurate vs f32 ground truth on 2-bit (its parity test pins no-worse-than); accuracy was never the problem.
- `dsv4_sink_softmax` (`MLX_SERVE_DSV4_SINK_SOFTMAX=1`): scale → sink-in-denominator → row-softmax → slice in one dispatch between the two attention GEMMs. Neutral-to-slightly-negative (~29.6 vs ~29.8) — the composed 4-dispatch chain was already overlapped (the fusedAttnGate on-chain-but-overlapped class), and the kernel changes greedy bytes (softmax tree). A byte-changing lever with no win defaults off. The full flash-style MQA-with-sink kernel (lever 4 proper) is hereby DE-PRIORITIZED on the same evidence: the two GEMMs are efficient and the glue between them is not on the visible critical path.

**Class lessons (new or re-confirmed):**

- **The QAT sims need NO transcendentals in-kernel**: scale = 2^ceil(log2(amax/code_max)) and the grid exponent floor(log2(|y|)) are EXACT via `metal::frexp`/`ldexp` bit arithmetic (frexp's mantissa ∈ [0.5,1) makes ceil/floor a comparison), matching the host f64 semantics rather than the composed chain's f32 log2/ceil approximations. The swigluSigTable rule still applies where a real transcendental remains: the gate+up kernel reads sigmoid from a 65536-entry f32 table built by mlx_sigmoid itself.
- **A `metal_kernel` arm owes its own `streamIsGpu` guard**: the dec-chain kernel wired into `attentionDecodeGpu` crashed 9 of 16 mini tests (exit 255, uncatchable) because the CPU-stream test arm reached it — the emission kernel had been accidentally shielded by `gpuEmitActive`. The guard belongs in the kernel helper, not the caller.
- **A per-token-varying TEMPLATE value is a fresh Metal JIT per value**: `TK` (attention key count, grows every token during the context ramp) baked as a template arg made request 1 run at 19 tok/s vs 29.7 warm — hundreds of one-off kernel compiles. Ramping values ride INPUTS (`tk_size` scalar); only stable geometry (H, TG) is templated. Corollary to the ShapeKey config rule: the config cache is not enough when the TEMPLATE varies.
- **Levers 1+2 change greedy bytes and that is sanctioned**: exact-vs-approx sim rounding + softmax/RMS reduction-tree drift the sims don't fully snap. The 6-prompt greedy characterization (2 identical, 4 equivalent-quality forks) gated default-on. A DSpark acceptance "regression" on the B-tree prompt (1.36 vs 1.86/round) evaporated across 4 prompts (means identical at 2.39): when an A/B's arms FORK CONTENT, per-prompt spec-decode cells are content variance, not a verdict — sample across prompts before believing one.

### Addendum: comp_in int8 requant — the characterization said opt-in (2026-08-01)

The parked lossy lever, gated behind the user-facing `--decode-attn-quant` config flag per its contract (decode-only lossy requant, prefill quality anchor). `comp_in_t` (the combined compressor-input operand, f32 BY CONSTRUCTION from `transposedF32` — the first eligibility check declined it as "not bf16" and the engagement test caught the silent no-op) gets int8-g32 side copies (362 MB, built eagerly at init inside the load budget) served at C ≤ 32: serial decode, DSpark verify blocks, and tiny suffix prefills — verify MUST see decode's weights or spec-on/off diverges (the laguna rule); big prefill chunks keep dense.

Measured: serial **29.4 → 31.6 tok/s (+7-8%)**, DSpark with the flag 44-47 tok/s at 3.0-3.2 accepts/round. On quality the sims snap MOST of the drift — 3 of 6 characterization prompts byte-identical, and the hash-table prompt was byte-identical through every arm — but the 7-prompt sweep found ONE real artifact: a duplicated opening paragraph on the B-tree prompt (the one that's already baseline-degenerate for this quant mix), which the dense arm doesn't produce. Answers moved → per the standing rule it ships **explicit-opt-in**: `transformer.decodeAttnQuantExplicit()` (CLI `--decode-attn-quant`, app toggle, or env `MLX_SERVE_DECODE_ATTN_QUANT=1`) engages it; the flag's silent default keeps dsv4 dense while laguna keeps its clean-characterization default-on. The general lesson: **a shared lossy flag's default is per-arch, decided by that arch's own characterization** — adoption is not inheritance.

Also from this arm: the wide characterization's first repetition detector strided 20 chars and MISSED the duplicated paragraph (a repeat only aligns with a strided window when the repeat distance divides the stride) — a repetition detector must stride 1 or it reads "clean" over real loops.

## Stochastic DSpark acceptance: sampled agent traffic gets the speedup (2026-08-01)

DSpark shipped greedy-only by design — `dsparkRoundWith` accepted drafts by raw argmax equality against host verify logits, so the chokepoint armed it only for `greedy_clean` requests. The checkpoint ships `generation_config` temp 0.6 and agent CLIs (pi, validated 2026-08-01) omit temperature, which meant **every real agent request ran serial (~27 tok/s with requant) while only a pinned `--temp 0` boot earned the 47-60 tok/s DSpark path**. The MTP stochastic round had already solved exactly this shape: filtered target probs at every verify position, Leviathan accept, pre-sampled residual corrections, ONE bounded sync.

**The port is small because nothing had to move**: `nextDspark` is a method on the same Generator that owns `probsAllPositions`, `mtpBatchedAcceptGraph` and `self.prng`. DSpark drafts are greedy stage argmaxes — one-hot proposals — which is precisely `mtpBatchedAcceptGraph`'s `q_probs == null` arm: accept draft k with prob `min(1, p_k)` (filtered target prob of the draft token), first reject at `a` corrected from `normalize(max(p_a − onehot, 0))`, full accept sampled from the bonus row. The output distribution equals serial sampling (the toy-vocab exactness test's invariant), and the correction still derives from the ORIGINAL verify logits at the acceptance point — the house partial-accept invariant in sampled form.

What dsv4 needed was a seam that hands the verify logits out LAZY and takes the accept decision from the caller:

- `extendChunk`'s comptime bool became a mode enum `{last_host, all_host, all_gpu}`; `.all_gpu` returns the un-synced `[C, vocab]` logits via `headLogitsBatchG` (the lazy half of the old `headLogitsBatchGpu`, which is now a 3-line wrapper = lazy graph + `toHostF32`). `.all_host` is therefore `.all_gpu` + host read BY CONSTRUCTION — pinned by a parity test that runs both arms from one snapshot on both streams and demands byte equality.
- `dsparkBegin`/`dsparkBeginWith` (draft → snapshot → anchors → lazy verify) and `dsparkFinish` (rollback-or-disarm → tokens → counts) split the old round; `DsparkPending` carries `{snap, vl_g, b, verify ids, phases, clock}` — NOT the draft, because the ids in `verify[1..b+1]` are all any accept loop needs. The greedy `dsparkRoundWith` was rewritten ON TOP of the split (begin → toHostF32 → the unchanged argmax loop → finish) so there is ONE verify seam and the greedy path can't drift from the stochastic one. Byte-pinned through the whole mini suite + test_dsv4.sh 14/14 + template 17/17.
- The chokepoint gate became the pure `Generator.dsparkArmFor(sampling, logprobs_n, stoch_enabled)`: `clean` (no penalties/grammar/logprobs — those stay serial on BOTH arms, matching the old contract) × `greedy` (temp<0.01 or top_k==1). Clean+greedy → argmax arm; clean+sampled → stochastic arm unless `MLX_SERVE_DSV4_DSPARK_STOCH=0`. Log lines distinguish the arms (`spec=dspark (stochastic; …)`), and the mini engagement test honors whatever env the test binary was launched with, so the `=0` run tests the fallback instead of skipping.

**Profiling note**: with begin returning lazy logits, extendChunk's head lap measures BUILD only — the head eval lands in the consumer's sync, so `DsparkPending.lapVerify` (called after that sync) owns the verify lap and `verify_head_ns` reads ~0 under the split. `verify_ns` still covers the whole verify; only the head sub-attribution moved.

**The deciding A/B** (paired boots, same box, `--dspark` + NO `--temp`, prompts omit temperature → generation_config 0.6; arm proven by engagement lines in its own log — `spec=dspark (stochastic): 4` vs `spec=disabled: 4`):

| prompt | stochastic | serial | speedup | accepts/round |
|---|---|---|---|---|
| code (CSV parser) | 54.8 tok/s | 26.9 | 2.04x | 3.98 |
| prose (Rayleigh) | 29.4 tok/s | 27.0 | 1.09x | 1.67 |
| echo-edit (docstrings) | 62.0 tok/s | 26.9 | 2.30x | 4.25 |
| qa-list (primes) | 57.4 tok/s | 27.3 | 2.10x | 3.94 |

Acceptance at temp 0.6 sits close to the greedy rates (3.9-4.3 on draftable content vs greedy's 3.4-4.5) because accept prob = filtered p(draft) and the model is confident exactly where the stages draft well. Low-draftability prose is the floor at 1.67 accepts (2.67 committed/round against the ~3.2 break-even) and still lands at serial parity, not below — so the default is ON, kill-switched. Outputs on both arms coherent and correct (stride-1 repetition detector clean; primes sum right, CSV escaping right).

Expectation to keep: the stochastic arm's speedup is CONTENT-DEPENDENT with a floor at ~serial parity. If prose-heavy workloads ever measure below serial, the follow-up levers are a smaller block size and the shipped-off confidence gate (`MLX_SERVE_DSV4_DSPARK_CONF`) — economics in the confidence-gate rule above.

**pi end-to-end** (the motivating case; isolated `PI_CODING_AGENT_DIR` config, NO `--temp` on the boot): a multi-round fizzbuzz+selftest task completed and verified independently (re-ran the test outside the agent), every request took the stochastic arm (3/3 in the server's own log), acceptance **2.8-5.0 accepted drafts/round** — one request full-accepted its entire block every round. Agent code traffic is even more draftable than the synthetic prompts; the greedy-only gate was leaving the best workload on the table.

### Addendum: the prefill guard's CHUNK term is arch-owned too (2026-08-01, the class's third bite)

Minutes into the first real pi session on stochastic DSpark, a 7514-token prompt 400'd: "requires ~4069MB but only ~3610MB available". Not pi's fault and not fixable with pi-side limits (7.5k is a normal agent prompt); with the stages resident the box really does sit at ~3.6-6 GB slack — but the bill was wrong, in both directions at once. The generic estimator charges `3 × mlp(chunk)` with `chunk` from `effectivePrefillChunk` (the 4096 MoE cap) — dsv4's `extendState` sub-chunks internally at `prefillSub()` (512), so the real MLP envelope is 8x smaller (~2.6 GB of phantom bill). Meanwhile the fp16 score-scratch term (42 MB) misses the arch's REAL attention transient: the `[C, tk, latent]` f32 gathered-K set (~670 MB — the very allocation PREFILL_SUB exists to bound). `server.dsv4PrefillMemoryNeeded` now bills f32 module-owned state (raw latents + ~half again for the compressed arms; kv-quant never applies here), the f32 gather, and the sub-chunk MLP — ~2.3 GB for the failed request, admitted. It reads the LIVE `dsv4_mod.prefillSub()` (env-overridable — billing a stale constant while `MLX_SERVE_DSV4_PREFILL_SUB` raises the width would under-bill into an uncatchable OOM), and the call site is scan-pinned with `++`-split needles so the scan test cannot match its own source. Live-verified: an 11.2k-token prompt prefills clean with the stages resident. Physics unchanged: DSpark's 11 GB of stages still caps agent prompts around ~12-16k on a 128 GB box (bill is kv-linear, ~132 KB/token) — the pi launcher config sets `contextWindow: 16384` so pi compacts before the wall; full-window 40k sessions want a serial (no `--dspark`) boot.

---


## Sparse video attention is dead by measurement, and the frames are what said so (MiniMax-H3, 2026-08-05)

Attention is 25% of an H3 step at 480p and 59% at 768p, and MiniMax deliberately
withheld their own sparse implementation from the release. So a training-free
SVG/STA-style pattern looked like the one large FLOP saving left on a DiT whose
every other component is already at the compute roofline.

It is dead. Not "needs tuning" — dead.

**The machinery is fine.** Per-layer spatial (within-frame) / temporal
(same-patch stripe) attention, with the GLOBAL STRIP (text/cond/audio rows)
concatenated into every tile's key set and strip queries keeping full attention,
dense anchor layers at the ends and middle. Pure mlx ops — batched SDPA with the
pattern axis folded into batch, no custom kernel. The subset selection is PROVEN
by a uniform-score closed-form test (q = k = 0 makes each output row the mean of
exactly its subset, so the test reads the pattern directly rather than trusting
it).

**Three numbers, in the order they arrived, and only the last one mattered.**

1. The µbench said **15x**. It was measuring the attention op in isolation at
   the sparse shapes, which is exactly what it was asked to measure.
2. Live, in the step graph, it measured **1.33x**. Same class as the lm_head
   prune and the fused QK-norm+RoPE: an isolated kernel win that the surrounding
   graph does not pay out, because the GPU was already overlapping the work and
   because attention shares the step with linears that did not get faster.
3. The rendered frames are **visibly smeared tiles**.

**The quality failure is structural, not a tuning miss.** The temporal arm sets
`nb = f` — one "batch" entry per frame — so on those layers two adjacent SPATIAL
positions are in different batch entries and never exchange information at all.
A video DiT cannot reconstruct local spatial structure from the dense anchor
layers alone, so the tuning ladder everyone reaches for (denser anchor cadence,
spatial-only sparse layers, SVG-style per-head online classification) is being
asked to repair a hole the pattern puts there by construction. At a 1.33x
ceiling it is not worth finding out how much of it can be repaired.

**Two instruments pointed the right way before anyone looked at a frame** and
both were treated as soft: PSNR vs the same-seed dense run was 9.8 dB, and the
sparse clip x264-compressed at **10x** the control's bitrate — the classic
added-noise tell, since noise is what a video codec cannot compress. The eyeball
pair (`h3_sp480q` vs `h3_d480q`) is what settled it, which is the same precedent
as every other quality call on this backend: **metrics flag, clips decide.**

### Corollary: no kernel work is justified on this DiT at all

Measured the same session, and it retires the whole category rather than one
lever:

| what | effective | ceiling |
|---|---|---|
| SDPA at `[1,56,9266,128]` | ~13.3 TFLOPS | ~15 TFLOPS dense bf16 |
| linears under dq-gemm | ~13.6 TFLOPS | ~15 TFLOPS dense bf16 |

Both are within ~12% of the machine's dense-bf16 roofline, so a *perfect* kernel
is worth ~15% of the step and a realistic one is worth single digits. bf16
weights buy 6.5% for 2x the footprint — declined. Wall-clock on this backend now
only moves by doing FEWER forwards (steps, step cache, attention broadcast), not
by making a forward faster.

Also found while measuring, and worth more than the sparse result: **the fast
recipe's gate was a NO-OP at <= 6 steps.** `attnBroadcastWarmup` + the tail
always-refresh window together cover the entire schedule at small step counts,
so every few-step run was silently getting the dense path while being credited
with the recipe. A gate whose windows scale with the schedule needs a test at
the SHORT end of that schedule, not only at the 30 steps it was tuned on.

## The near-repeat loop tier convicted a voxel scene (2026-08-05)

The tier shipped on 2026-08-04 to catch a loop that rephrases itself. One day
later it destroyed a legitimate generation: asked for "a very creative,
elaborate, and detailed voxel art scene of a pagoda in a beautiful garden", a
pi session was cut at **16241 generated tokens** and wrote no file at all.

The output was fine. Its tail was `fillBox(-5, 7, -5, 5, 7, 5, C.orangeTile);`
lines with fresh coordinates on every one. The problem is that both of the
tier's ratios measure a VOCABULARY, and procedurally generated scene code has a
loop's vocabulary by construction: one call template plus a small colour
palette. Measured on the artifact itself: distinct-token ratio **0.068** against
a 0.12 bar, distinct-4-gram ratio **0.351** against a 0.35 bar — it escaped
conviction by 0.001, and the generation's own tail did not escape.

The separator is PROGRESS, not vocabulary. A loop stops introducing material; a
scene keeps adding geometry. Measured as "what fraction of the window's
second-half 4-grams did its first half never contain":

| content | tokens | 4-grams | novelty |
|---|---|---|---|
| the cut voxel artifact | 0.068 | 0.351 | **0.632** |
| dense procedural scene code | 0.021 | 0.304 | **0.298** |
| a markdown table | 0.016 | 0.541 | **0.827** |
| restatement loop | 0.024 | 0.052 | **0.019** |
| file-repair restatement loop | 0.041 | 0.092 | **0.022** |

Two orders of magnitude apart, so the bar is 0.10 — ~4.5x above the loops,
~3x below the closest healthy case. All three ratios must now be low.

Three things worth keeping:

- **The direction of the error matters.** A missed loop still ends at
  `max_tokens`; a false cut destroys work that was going fine and, on a tools
  request, returns nothing at all. Every ambiguity in this tier resolves toward
  acquittal.
- **The known miss is honest.** A restatement loop that carries a changing
  counter ("Attempt 1…", "Attempt 2…") scores 0.651 and is acquitted — by this
  measure it IS progressing. Catching it needs a different signal, not a lower
  bar.
- **The regression fixture is derived, not invented.** The first attempt at it
  interleaved template and varying tokens and did NOT convict, i.e. it was a
  test that could not fail. The shape that reproduces the bug puts the template
  tokens in a CONTIGUOUS run followed by the coordinates, so most 4-gram windows
  sit entirely inside the fixed run and repeat every line (0.033 / 0.316). The
  parameters were swept numerically against the measured artifact before being
  written into the test.

## A prefix-cache hit is bit-exact on attention and is not on a hybrid (2026-08-07)

Surfaced by llmprobe's determinism check during 26.8.3 pre-release validation:
`greedy runs diverged at char 275 (det-echo-sentence, 3 runs) — non-determinism
at temperature 0`, on LFM2.5-2.6B-8bit.

**The shape of the evidence.** Four identical temp-0 requests, same boot:

| arm | run 1 | runs 2-4 |
|---|---|---|
| defaults | A | B B B |
| `--no-pld` | A | B B B |
| `--no-pld --prefix-cache-entries 0` | C | C C C |
| `--prefix-cache-entries 0` | C | C C C |

Run 1 is cold, runs 2+ are warm. `--no-pld` changes nothing, so it is NOT spec
decode — which is the first thing everyone suspects, and the reason to run the
kill-switch A/B before theorising. Turning the prefix cache off makes every run
identical. Note there are THREE distinct outputs, not two: the cold-with-cache
answer also differs from the no-cache answer, because merely enabling the cache
changes the prefill shape (the SSM snapshot sits `SSM_SNAPSHOT_BACKOFF` = 30
tokens before the prompt end, splitting the pass).

**Is it corruption or rounding?** Answered with the logprobs, not by eyeballing
the text — which is a good use for the field the same release had just fixed:

```
first divergence at token index 44
  prefix: ' why determinism matters for inference engines. I must follow the'
  cold: chose ' instructions' lp=-1.242188  rank1-rank2 gap=0.125000
  warm: chose ' rules'        lp=-1.335938  rank1-rank2 gap=0.000000
pre-divergence logprob agreement over 40 tokens: max|delta|=4.6875e-02  mean=2.76e-03
```

The warm path shows an EXACT tie between the two candidates. The two paths
agree to a mean of 2.8e-3 for the whole run up to that point, with a worst case
of 0.047 — which is exactly bf16 rounding scale (~8 mantissa bits on logits of
magnitude 10-20). A restore defect smears the whole distribution; this does
not. Not corruption.

**Mechanism.** For a pure-attention arch, a warm restore replays STORED KV
values — nothing is recomputed, so it is bit-exact. gemma-4-e4b-it-4bit through
the identical probe: `NO DIVERGENCE over 220 shared tokens`, with a full
38-of-38-token reuse. A hybrid instead restores the conv/SSM state at the
checkpoint position and RE-RUNS the recurrence over the remaining prompt — the
cold pass ran positions 0..40 as one block, the warm pass runs 10..40, and a
sequential recurrence in a different block size is the same mathematics in a
different reduction order.

**Not a storage bug.** `captureSsmCheckpoint` copies through
`materializedOwnedCopy`, which preserves dtype; there is no narrowing anywhere
on that path.

**Why the existing guard is green.** `test_hybrid_reuse_equivalence.sh` asserts
warm output is byte-identical to cold and passes on this very model — its
prompt is 1391 cached tokens and its generation happens to land on no near-tie.
The exposing shape is the opposite: a SHORT prompt (so the 30-token backoff is
a large fraction of it) with a LONG generation (so the run has many chances to
hit a tie). A byte-equality test over one prompt samples one path through the
distribution; it cannot stand in for a determinism claim.

**Standing rule.** Byte-stable long greedy on a hybrid requires
`--prefix-cache-entries 0`, exactly as it requires no-spec + `--kv-quant off/8`
for the INT4 near-tie class. Both are properties of the arithmetic, not bugs to
fix, and both belong in any determinism claim we make.

## One request killed the server: a borrowed-handle cache that freed a live entry (2026-08-07)

Found during 26.8.3 pre-release validation, by the conformance sweep rather
than by any test — and it had already corrupted a benchmark row nobody would
have questioned.

**How it presented.** The `qwen3_5_moe` family row scored **33.3% engine
conformance**: `responses 0/26`, `messages 0/27`, `completions 0/1`, and
**0% model capability** with every knowledge question 0/1 — a model that
apparently could not name a chemical symbol. That reading is the tell. Whole
surfaces at exactly 0/N, plus a capability floor of 0, is not a weak
checkpoint; it is a **dead process**. The server log confirmed it: 13 requests
where every other cell logged 34, ending mid-request with no shutdown line.

**Two crash reports, one function.**

```
SIGSEGV  mlx::core::array::~array < mlx_vector_array_free
                                  < verifyQmm < qmatmulBits < qmatmul
                                  < gatedDeltaNet < forwardMoeWith
                                  < Generator.nextPld      (decode / PLD verify)

SIGBUS   vector<array>::__emplace_back_slow_path < mlx_vector_array_new_data
                                  < verifyQmm < ...
                                  < Generator.initWithOptions  (prefill)
```

**Root cause.** `cachedScalarInt` caches 0-d int scalars for the kernels' K/N
inputs in 16 slots, and evicted slot 0 unconditionally:

```zig
_ = mlx.mlx_array_free(vqmm_scalar_cache[0].arr);
vqmm_scalar_cache[0] = .{ .v = v, .arr = arr };
```

All six call sites do this:

```zig
const K_arr = cachedScalarInt(K);
const N_arr = cachedScalarInt(N);
const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
```

Once the cache is full: the K call evicts slot 0 and lands there; the N call
frees slot 0 again — now K's array — and `inputs_arr` carries a freed handle
into mlx.

The code carried a comment asserting this was safe: *"in-flight lazy graph
nodes hold their own refs to the old array; this only drops OUR handle."* That
is true, and it is irrelevant. The array has not reached a graph node yet. It
is sitting in the caller's stack array, one line above.

**Why it stayed hidden.** Eviction only happens after **more than 16 distinct
(K, N) values**. Most checkpoints never get there. A 40-layer MoE with
GatedDeltaNet, MoE expert widths and a 248k vocab does. Nothing about it is
model-specific — any checkpoint with enough distinct geometries is exposed, and
it fires on both the prefill and decode paths.

**Fix.** LRU eviction (`vqmmScalarEvictIndex`): pick the entry with the oldest
tick. Correct by construction — the scalars a call site is still holding are by
definition the two most recently ticked, and the cache is 16 while the most any
site takes is 2, so they can never be the minimum.

**Before / after, same command:**

| | pre-fix | post-fix |
|---|---|---|
| engine conformance | 33.3% | 99.5% |
| chat / responses / messages | 34/54, 0/26, 0/27 | 65/65, 62/62, 60/61 |
| model capability | 0% | 97.2% (strong) |
| server | dead | alive |

pi agentic tasks on the same checkpoint then ran 3/3 with PLD engaging 12
times — `nextPld → verifyQmm` being the exact path that segfaulted.

**Two lessons worth keeping.**

1. *A comment asserting safety is not safety.* This one was written
   confidently, was half-right, and cost a server-killing crash. The question
   it answered ("does the graph hold a ref?") was not the question that
   mattered ("has it reached the graph yet?").
2. *A bench harness reports whatever the engine returns.* llmprobe measured and
   charted a decode rate for a process that had already crashed. The corrupted
   row's numbers looked ordinary. What gave it away was structural — its log
   was a third the length of every other cell's, and it had no shutdown line.
   Worth checking cell logs for uniformity before trusting a bench row.

## The prefill-chunk cap must read the width the SCORE is contracted at (`prefillScoreHeadDim`)

An arch can score at a different width than it stores values at (an MLA q·k contracting over nope+rope while `head_dim` stays the value width), and `boundedPrefillChunk`'s `<= 128` "fused SDPA covers it" early-out then silently exempts exactly the arch whose composed path materializes the biggest score tensors. `ModelConfig.prefillScoreHeadDim` is the width the cap must read. The two hd-256-MEASURED policy branches (the fused-kernel early return, the 2048 composed cap) stay keyed on 256 EXACTLY; neither generalizes. A new arch scoring wider than it stores adds its arm to `prefillScoreHeadDim`. A cap that trades throughput for reach is the usual expectation; measure before assuming you are paying for it.

## A per-arch spec exclusion written as a hand-rolled conjunct is a list of ONE (2026-08-05)

The dsv4 fixes both named dsv4 IN PLACE — `modelExclusiveDecode` returned `t.dsv4 != null`, and runPrefill spelled `!is_dsv4` on each of the three `use_mtp`/`use_drafter`/`use_pld` lines. A second module-owned arch arrived with the SAME shape (module-owned `Model.state`, `reset = ctx.cache.step == 0`, a 0-layer shell `KVCache` so snapshot/truncate are no-ops) and got neither: two concurrent requests shared one state, and `--pld` drove verify forwards straight through it with nothing to roll back. Both now read ONE predicate — `Transformer.ownsModuleDecodeState()` over the named `module_owned_state_fields`, and `scheduler.specInitWiring` (pure, `owns_module_state` in / the three use_* + a `native_intent` bit for an arch with its OWN draft mode out). Guards: a class scan asserting every `?*<x>_mod.<Y>` field on Transformer is in the list, delegation scans on both call sites, and `specInitWiring`'s table.

Corollary 1 — the exclusion is per-arch CAPABILITY, not ownership (`Transformer.moduleStateSpecRollback`): an arch earns spec modes by shipping per-position state capture plus a rewind, pinned against fresh forwards at EVERY accepted count (bit equality is the wrong bar — a verify projects 1+m rows in one matmul, a reference one at a time, and MLX picks its GEMM by M; assert the error RATIO between correct and off-by-one instead). dsv4 cannot roll back (rings/compressed caches have no per-position capture) and keeps every shell spec mode off; it has DSpark for drafting.

Corollary 2 — a module-owned forward that ignores `ctx.capture_hidden` hands spec decode an EMPTY hidden: `forwardWithCaptureAll` only sets the ctx fields and calls `forwardWith`, so an arch that never reads them silently captures nothing — and the MTP round then drafts from a null handle rather than failing. The fix is the `skip_lm_head` seam: run the trunk without the head, publish last-row + all-rows, project afterwards. (Found on the since-removed ling port; the wiring obligation stands for any new module-owned arch that wants spec decode.)

## Kokoro port: the SineGen no-op, load/layout traps, and the vocab invariant

**Reproducing an upstream NO-OP faithfully means NOT doing it**: the reference adds a random initial phase at SAMPLE 0 then linear-downsamples by 1/300 with `align_corners=False`, which reads position 149.5 — so the offset is never read and the round trip is bit-identical with and without it (verified max abs diff 0.0). This port collapses that identity round trip and works at the frame rate, where adding it is REAL. Cost: waveform cosine 0.477 instead of 0.997. Caught only by measuring the reference's OWN seed-to-seed self-similarity (0.9941–0.9960) instead of accepting "it's a stochastic vocoder" as licence for a loose threshold — a stochastic component excuses thousandths, not halves. Any port that simplifies a reference's resample/round-trip owes that measurement.

**Load/layout traps**: safetensors READ on the CPU stream (mlx `Load` has no GPU impl — uncatchable `[Load::eval_gpu] Not implemented`); a DEPTHWISE `ConvTranspose1d` transposes `{0,2,1}`, NOT `ltx_audio`'s groups=1 `{1,2,0}`; mlx-c has no "build a complex array from two reals" op, so the iSTFT spectrum is `re + im·i` via `mlx_array_new_complex`; `AdaIN1d` declares `affine=True` but the checkpoint ships NO `norm.weight/bias` (loaded `strict=False`), so it is pure instance norm — read the CHECKPOINT, not the reference source.

**Vocab**: Kokoro's vocab includes uppercase ASCII as diphthongs (A I O W Y Q S T), so "no ASCII letters in the phonemes" is a WRONG assertion — `həlˈO` is correct. The real invariant is that every emitted symbol is IN the vocab, because `Vocab.encode` silently DROPS unknowns (an unexpanded number just vanishes from the speech rather than erroring).

## The DFlash assistant's own precision is a per-round read, and the A/B that measures it must not name its arms (2026-08-10)

Shipped DFlash reads the WHOLE assistant every round: one block forward over 5.11 GB of bf16, plus the trunk lm_head twice (draft argmax over `block−1` rows, then the verify's own read of the same 202048×6656 head). Neither read is a fidelity question. The trunk verify is exact whatever the drafts were, so the only thing a lossier draft path can cost is ACCEPTANCE — which is a number you measure, not a risk you argue about.

So both reads got shrunk. `dflash.DflashLinear` is one handle with two arms and one `apply`: dense bf16 pre-transposed to `[in, out]` for a plain matmul (what v1 did), or affine-packed `[out, in]` served by `mlx_quantized_matmul(transpose=true)`. The packed layout is the CHECKPOINT's layout, so quantizing at load REPLACES the dense arm's pre-transpose rather than adding a step — the load does strictly less work and the resident weights halve at 8 bits. Three rules fell out of doing it per weight rather than per model: a sidecar that already ships `.scales` is served as-is at the width `affineParamsFromGeometry` solves from its packed geometry (asking for "dense" must not unpack it — a request for a precision is not a request to re-encode someone else's artifact); a weight whose contraction dim divides neither 64 nor 32 stays dense on its own (`quantGroupFor`), because affine group sizes are 32/64/128 and a model-wide yes/no would refuse a whole sidecar over one odd projection; and `bits == 0` is the dense discriminator, never a probe of the scales handle.

The draft-only lm_head is the MTP trick, unchanged: re-encode the trunk head once at bind, project drafts through it, leave verification on the trunk's. Two things it taught. `mtp.requantizeRows` already had the chunking discipline (dequantize a row chunk, requantize, eval, concatenate — never materialize the whole 2.5 GB dense head), so the right move was extending it with a DENSE-source arm rather than writing a second requantizer in dflash.zig; a chunking policy that exists twice drifts. And the resolvers that name a head's true params dereference the scales handle, so the DENSE arm has to be decided BEFORE calling one — `computeQuantParams` on a dense head's null scales is not a wrong answer, it is `expected a non-empty mlx_array` from mlx-c and a dead test binary. mtp's own `buildDraftHead` returns early on exactly that check; the dflash copy computed the params first and crashed, which is the same bug the early return was hiding.

**The measurement lesson is the durable one.** Load-time quantization is a BOOT decision, so the arms cannot share a boot, and the first cut of the harness gave each boot a prompt nonce that named its arm (`[r1 dense t0.0] …`) to defeat the prefix cache. Every arm then diverged from every other on every prompt — because the model is always-thinking and restates the instruction, so one differing byte in the prompt re-rolls the entire generation. That does not merely add noise: it means the arms are not running the same work, the tok/s columns are comparing different token streams, the acceptance columns are comparing different content, and the greedy-identity check that should PROVE the drafts never leak into the output instead reports 18/18 divergent for a reason that has nothing to do with the change. A nonce must vary by REP and be identical across arms; a fresh boot starts with an empty prefix cache anyway, so the nonce is only ever defeating within-boot reuse. Same class as "an A/B arm is proven by ENGAGEMENT lines in its own log, never its launch env": a harness that cannot tell you the arms did the same work cannot tell you which arm was faster.

## DFlash perf round 3: the block is a GPU decision, and it was worth 2x (2026-08-10)

DFlash shipped at the assistant's config block of 16 and was worth 1.01x on a
4-bit Muse trunk. `MLX_SERVE_DFLASH_TRACE=1` (added for this; it inserts eval
barriers, so a traced round is slower than a real one) split the round at block
16: verify 151 ms, assistant 14.2, draft head 10.2, append 1.1, scheduler gap
0.01 — against a 35.8 ms serial step. The verify reads the same weights as a
serial step and cost 4.2 of them, because it ran 16 rows wide. That is also why
halving the trunk never moved it: compute-bound, not read-bound.

MLX picks a transposed-qmm kernel by `get_qmv_batch_limit(K, N)` (10 on these
shapes) — below it the weight-reuse `qmv`, at/above it a 32x32-tiled
`qmm_splitk` that wastes half a tile at width 16. Our split-K lane covers M 2..7
and the NAX m16 tile covers M 8..16 on G17 only, so 8..16 on a G16 has no lane
at all. Whole-forward µbench per width (`MLX_SERVE_VERIFY_WIDTH_UBENCH=1`, one
eval per BATCH of launches — a per-launch barrier is ~0.14 ms of sync here,
more than the whole kv projection), calibrated against the live 35.8 at M=1:

```
 M    1     2     3     4     5     6     7     8    10    16
ms  33.0  35.9  34.9  38.5  42.9  47.0  55.7  84.4 138.1 138.4
```

Live, serial reference in the same boot: block 16 → 0.92x, 8 → 1.16x, 7 → 1.43x,
6 → 1.79x, **5 → 1.97x**, 4 → 1.81x, 3 → 1.77x. The default now caps at
`NO_WIDE_LANE_BLOCK_CAP` when `wideVerifyLaneAvailable()` is false. Not a
monotone win — a smaller block drafts fewer tokens, so 3 is worse than 5.

**The assistant context has to ride the prefix cache.** A restore forwards no
trunk layers, so it produces no `capture_layers` output, so `DflashCtx` started
empty on every reused prefix — 92.6% → 66.5% per-draft and 80.2 → 60.9 tok/s on
a full-prefix hit, i.e. multi-turn paid for the cache twice. Fixed by
snapshotting it onto the entry (`prefix_cache.DflashSnap`), bytes folded into
`kv_bytes`; hits now report cold-identical acceptance. Cheap to get right
because it is DRAFT-side — every failure path (no payload, disk tier, misaligned
base) returns `dflash_base = null` and starts blind rather than erroring. The
adopt gate is `base + step == matched` EXACTLY, since `nextDflash` asserts it.

### Levers measured and closed

| lever | verdict |
|---|---|
| draft-only 3-bit lm_head | **OFF.** A win at block 16 (10 ms of 175), a loss at block 5 (under 2 ms of ~60): 55.4 tok/s / 2.19 accepted per round vs 57.8 / 2.37 on the trunk head. Re-measure every draft-side lossy default after the round cost moves. |
| cross-round pre-draft | **Dead.** Traced scheduler gap is 0.01 ms; nothing to overlap. |
| prefill capture cost | **Free.** −5.2% / +2.3% / +0.3% at 1128 / 5525 / 16185 prompt tokens. |
| adaptive block size | **Declined.** The spread that justified it (creative peaked at 3, echo at 5) was an artifact of the 3-bit head; with it fixed every class peaks at 5. |
| assistant micro-opts | **Not worth pricing.** Halving its weight read outright (`QUANT_BITS=4`) is no faster at identical acceptance. |
| wider split-K tile (bn 8) | **5-8% slower** at every width, though the activation-traffic arithmetic says it should halve the dominant read — the x rows are 66 KB and never left cache. Kept behind `MLX_SERVE_VQMM_BN`. |
| k_parts | Shipped 2 is optimal (1: 42.5, 2: 42.4, 4: 44.5, 8: 49.0 ms). |

The split-K lane itself is worth +34% live on dflash and ~0 on serial — the
right shape for a verify-width kernel.

### Sampled drafts: clean theory, negative measurement

Greedy drafts accept through a one-hot q, so acceptance is `p(argmax q)`, which
a temperature-flattened target row should deflate. It does not. Greedy: temp 0 →
1.98x / 55.1% per-draft, temp 0.7 → 1.98x / **57.7%**. Drawing each draft from
the request's own filtered distribution and accepting through the full Leviathan
ratio is exact and a LOSS: 1.87x / 54.5%. Sampled acceptance is `1 − TV(p, q)`,
so matching the proposal only wins when q is close to p in SHAPE — this
5-layer sidecar picks the right top token far more often than it gets the tail
right. Left behind `MLX_SERVE_DFLASH_SAMPLED_DRAFTS=1`, default off.

Two lessons. **A single boot is not a measurement at temperature**: the first
run read 49.3% at 0.7 and looked like a clean 13% penalty; a second boot of the
same build read 54.0%. Sampled cells fork their own content and swing ~10% run
to run. And building the proposal for all m rows at once made `applyTopK` the
first caller ever handed more than one row — it used axis-less `mlx_topk`, which
FLATTENS, so the loudest row's cutoff masked the rest to -inf and softmax
returned NaN. Correct-by-accident for every `[1, V]` caller before it. A
reduction helper that has only ever seen one row cannot reveal an axis bug, and
"only ever seen one row" is a property of the CALLERS.

### The sliding trim was decode-only for every arch, and its guard predicate had drifted

`KVCache.updateDense`/`updateAffine` computed the trimmed view start under
`is_decode = new_len == 1`. Single-token decode got the last `sliding_window`
entries; **every other forward fell through with `view_start = 0` and read the
entire cache on every sliding layer**. That is every spec-verify block and every
prefill chunk. On muse-glimmer, which slides 39 of 52 layers, verify went 51 ms
at 0.6k to 140 ms at 17.6k and DFlash turned from a 2.28x win into a 0.74x
LOSS — the drafter was making things slower the longer the conversation got.

The fix is one line of arithmetic: a block of `q_len` queries needs
`window + q_len - 1` tail entries, because the FIRST query of the block reaches
back furthest, not the last. Decode is the `q_len == 1` case of the same
formula, which is why it looked correct for so long.

**Why trimming the view is safe at all.** Every consumer of the K view derives
its query offset RELATIVELY: `createCausalMask`/`createSlidingWindowMask` use
`offset_val = kv_len - q_len`, the fused hd-256 kernel uses `q_off = kL - qL`
with the band test `(row_pos - col) >= SW`, and `inklingAttnWith` already read
`kv_len` off the view shape and threaded `kv_start = total_kv - kv_len` into its
bias and mask. So a trimmed view is correct **provided the mask is built at the
trimmed length**. Hand a trimmed view a mask sized at `total_kv` and it does not
crash — it attends to the wrong slice and the output silently changes. That
pairing is the whole contract, which is why the guard is per-arch greedy
byte-identity rather than a shape assertion.

**The band exception, and the predicate that had drifted off it.** The one
consumer reading ABSOLUTE positions is the fused hd-256 kernel when it
band-masks in-kernel, so `band_in_kernel` declines the trim. That predicate
tested `seq_len >= 2` — but `fusedSdpa256Prefill` declines everything below 16
(`FUSED256_MIN_Q_LEN`, matching oMLX's `_MIN_ROUTE_Q_LEN`; short queries belong
to MLX's `sdpa_vector`). So every gemma spec verify, at 4-8 wide, was treated as
kernel-handled, declined the trim, and then fell through to the explicit-mask
path anyway. **Gemma got nothing from the trim at all.** A guard predicate that
is stricter than the thing it guards is a silent no-op: nothing fails, nothing
logs, the optimization just never runs. Both sites now read the one constant and
a source scan pins them together.

**The width cap was the other half.** A `SLIDING_TRIM_MAX_WIDTH` of 32 kept
prefill chunks out while the mask pairing was unproven. That is exactly backwards
for a SERIAL arch: laguna and inkling never speculate, so a prefill chunk is the
only multi-token forward they ever do, and the cap excluded the only case that
could have helped them. With `band_in_kernel` gating the real absolute-position
consumer, width needs no bound.

Measured, all same-session pairs (screenshare on the GPU, so read the shape not
the third digit):

| arch | what | trim off | trim on |
|---|---|---|---|
| muse-glimmer | verify @ 17.6k | 140 ms | 80 ms (DFlash 0.74x → 1.10x) |
| gemma4-26b-a4b | decode @ 18.7k, PLD | 95.5 tok/s | 102.9 tok/s |
| laguna-XS | prefill @ 19.4k | 504 tok/s | 763 tok/s |
| inkling-small | prefill @ 19.2k | 125 tok/s | 181 tok/s |

The gemma win grows monotonically with context (flat at 0.7k, +2.4% at 4.9k,
+5.5% at 9.6k, +7.7% at 18.7k) — the signature of a read that used to scale with
the whole cache and now scales with the window.

**Testing trap: a prefill chunk only takes the trim once it lands at a non-zero
offset.** Chunk 1 has `total_kv == seq_len`, which the span always covers, so
nothing is trimmed. With the default 8192 chunk a serial arch needs a prompt
past ~16k before any chunk trims — an 8k prompt prefills in ONE chunk and the
trim-on/trim-off arms agree for the wrong reason. `slidingViewFor` therefore
logs a one-shot `[sliding] block trim engaged` line, and
`tests/test_sliding_window_trim.sh` asserts its presence in the ON arm's log and
its ABSENCE in the OFF arm's — an A/B arm is proven by an engagement line in its
own log, never by the env it was launched with.

**Not covered here.** The diffusion canvas path uses a different convention
(`sliding_window - 1`, plus the whole canvas, which is not a causal tail), and
`createSlidingWindowDecodeMask` is now an all-zeros no-op — with the view
already exactly `sw`, `window_start = kv_len - window = 0` masks nothing, yet
passing `sel_mode = "array"` likely costs MLX's fused SDPA path. Both are real
follow-ups, neither belongs to this change. The prefill memory guards
(`prefillAttnKeys`, `server.dsv4PrefillMemoryNeeded`) still bill on `total_kv`,
so they now over-estimate — conservative, and retuning them is its own measured
change.
## KDA (Kimi Delta Attention): a lower bound that replaces the formula, and a gate that is an indexing contract (2026-08-11)

Ling 3.0 (`bailing_hybrid`) rides the existing GatedDeltaNet recurrence, and every place it diverges is a place a reasonable reading is wrong.

**The lower bound is not a clamp.** The config says `kda_lower_bound: -5, kda_safe_gate: true`, which reads exactly like "clamp the log-space gate at -5". It is not. fla's `fused_recurrent_kda_fwd_kernel` has two arms:

```
if USE_LOWER_BOUND:  b_gk = lower_bound * sigmoid(exp(b_A) * b_g)
else:                b_gk = -exp(b_A) * softplus(b_g)
```

The bound SELECTS a different function — a bounded sigmoid in `(bound, 0)` — rather than truncating the softplus one. Both are monotonic and both saturate, so a clamped-softplus port produces plausible text and would likely never be caught by eyeballing output; the two disagree most in the middle of the range, where almost every token lives. The rule generalizes past this arch: when a config key looks like a bound on a formula you already implement, read the reference KERNEL's arms before assuming it modifies yours. `kdaGateChain` is pinned by the two properties a clamped-softplus reading breaks — the decay never reaches the floor `exp(bound)`, and it is monotonically DECREASING in `a` (the softplus form's magnitude is unbounded there).

**The gate's shape is an indexing contract.** GatedDeltaNet gates per (token, value head); KDA gates per KEY CHANNEL, so `g` is `[B,T,Hv,Dk]` and each state element decays by its own factor. That is three lines of difference in a 40-line Metal kernel (the base pointer, the subscript, the per-step advance) — which is exactly the kind of difference that gets copy-pasted into a second kernel and then drifts. `gdnKernelSource(vectorized, capture_seq)` generates all four variants (scalar/vector × plain/spec-capture) from ONE recurrence, so the delta rule itself exists once; the existing seq-vs-single parity tests characterized the refactor.

Two things fall out. First, a kernel written for the OTHER shape must decline rather than read the wrong element: the blocked oMLX prefill kernel takes a per-head gate, so it is refused outright on a vector-gated arch (a perf loss on 18 of 24 layers, not a correctness risk — noted as a follow-up). Second, the equivalence test is a per-channel gate held UNIFORM across each head, which must reproduce the scalar kernel EXACTLY (same recurrence, same reduction order, only the fetch differs). A wrong stride — reading head 0's channel for every head, or advancing by `Hv` instead of `Hv*Dk` — passes every "output is finite and roughly decays" check and fails this one. The test also asserts the reference state is non-zero, because an all-zero state would pass any diff.

**A per-variant kernel cache needs a comptime capture.** `getGdnKernelFor(vectorized, capture_seq)` held its compiled kernel in a `struct { var kernel = null; }` declared inside the function. Zig memoizes a struct type declared in a function body when it captures nothing from the comptime parameters — so all four instantiations shared ONE cache slot and the first kernel compiled was handed back for the rest. The symptom was the seq-capture parity test exiting 255: it asked for three outputs and got a kernel that declares two. Adding `const is_vectorized = vectorized;` (and its twin) to the struct forces four distinct types. Any "one cached thing per comptime variant" needs the same capture.

## Follow-ups from the Ling 3.0 review: a cache that assumes one width, and a gate arm that must be chosen (2026-08-12)

Three holes the `bailing_hybrid` port left open, all of the same shape — a generalization applied to one code path and not its twin.

**`updateDense` learned that K and V can differ; `updateAffine` did not.** The dense write path was generalized to read each buffer's own head dim, and the MLA comment noted that "the quantized-KV fused kernels assume one head width for both, so MLA never opts in". That is true of the fused READ kernels and says nothing about the cache SCHEME: `--kv-quant 4|8` builds an `.affine` cache for whatever arch is loaded, and `updateAffine` solved `q_last`/`sc_last` once from `new_k`'s head dim and handed them to all six `growQuantBuf` calls. At K 192 / V 128 the value buffer came out 24 u32 wide while `new_vq.q` was 16, so `writeAtOffset` slice-updated a narrow chunk into a wide window — an mlx-level shape error, i.e. one we cannot catch. `truncate`'s affine arm had the same bug in view form (value scale/bias views sliced against the KEY scales' shape). The rule: a "K and V may differ" generalization is not done until every buffer in every scheme is sized from the operand it stores. Both are now covered by `KVCache affine quant carries an asymmetric K/V too`.

**A lazily-built rotation refuses lazily.** TurboQuant needs a power-of-two width and MLA's key is 192, so it genuinely cannot serve this arch. But `TurboState` builds its Hadamard matrices at the first write, so the refusal (`error.NonPowerOfTwoHeadDim`) fired inside the first request that reached an MLA layer — a 500 mid-generation for something knowable at load. `initWithConfigAndHeadDim` had even taken a `head_dim` parameter and thrown it away (`_ = head_dim; // observed at first write`), while its own doc comment claimed "the scheduler's load path validates head_dim is pow2 at cache init". The parameter is now used — against `ModelConfig.kvCacheKeyHeadDim()`, because `head_dim` says 128 for an arch that caches 192-wide keys, so the declared field would have validated the wrong number. When a lazy constructor's constraint is knowable from config, check it eagerly and let the request path keep the lazy build.

**Two arms means the arm is selected, not defaulted.** `kda_gate_lower_bound` was declared "0 = plain -exp(A_log)·softplus form" and the forward then handed that 0 straight to the bounded chain, which computes `exp(0 · σ(…))` = 1: a forget gate that never forgets, on a checkpoint whose only difference was omitting the key. The parse refuses a non-negative bound that is PRESENT, but an absent one left the field at its default and the field's own comment unhonored. The fix is a predicate (`ModelConfig.kdaUsesBoundedGate`) both sides read, and the absent case routes to the softplus chain — which is elementwise, so with `A_log` already expanded per key channel it serves a per-channel gate with no new code. Whenever a config value's default means "the other formula", the selection belongs in a named predicate; a default that silently degenerates one branch is worse than either branch.

## The refusal was right and the process died anyway: a fallible re-init behind a deinit (2026-08-13)

Making `KVCache.initWithConfigAndHeadDim` fallible was the previous section's fix — TurboQuant cannot serve a 192-wide MLA key, so say so at load instead of mid-request. It worked. `mlx-serve --model <Ling-3.0-tiny> --serve --kv-quant turbo4` printed the named refusal, correctly, and then died:

```
--kv-quant turbo: cache key width 192 is not a power of two ...
EXC_BAD_ACCESS in mlx_array_free <- transformer.freeKVEntry <- KVCache.deinit
  <- Transformer.deinit <- scheduler.doLoadOnInferenceThread
```

The crash is not in the new check and not in the arch. Every site that re-applies a kv-quant config was written when the constructor was infallible:

```zig
xfm_ptr.cache.deinit();
xfm_ptr.cache = try KVCache.initWithConfigAndHeadDim(...);
```

Read that with an error in mind: `deinit` frees the entries slice and every mlx handle in it, `try` returns, and a FREED cache stays installed on the Transformer. The owner's own `deinit` then walks the same entries and frees all of it a second time. Four sites had the shape — the scheduler's cold load, main's offline path, `resetCache`, `tryRestoreCache` — because the pattern is the obvious way to write it and was correct for as long as the callee could not fail. **A constructor becoming fallible is a change to every caller that frees before calling it**, and nothing in the type system says so: the `try` was added at each site by the same patch that introduced the error, which is precisely when the freed-object window opened.

The fix is ordering, held in one place. `KVCache.reinit` builds the replacement, and only then frees and swaps:

```zig
const fresh = try initWithConfigAndHeadDim(self.allocator, num_layers, config, head_dim);
self.deinit();
self.* = fresh;
```

Failure now leaves the live cache exactly as it was — same scheme, same contents, still serving — which is also the behavior a 503-and-retry story needs, since the alternative to crashing was a model entry holding a cache that could never be used again.

Two things make this a class rather than a bug. First, the symptom is maximally misleading: the stack names `mlx_array_free`, several frames and one full load-path unwind away from the refusal that caused it, so the natural first suspect is the MLA cache geometry or mlx's refcounting — anything but the four-line caller that printed the correct error message immediately before. Second, the shape recurs on its own: any future scheme that refuses an arch, at any of these sites, reopens it. So the guard is a source scan (`.cache = try` appears nowhere in transformer/scheduler/main, and `reinit`'s build precedes its free) rather than only the behavioral test, and the behavioral test is written against the testing allocator so the pre-fix order trips a double free on its own `defer` — verified red exactly that way, and red again with the helper's two statements transposed.

The general rule: **when a fallible call sits after a `deinit` of the thing it replaces, the error path is a use-after-free.** Build, then swap — and when several call sites share the pattern, the ordering belongs inside one helper they all use, not repeated correctly four times.

## The verify-qmm plain-SIMD tiles were a 4-bit specialization, and shape-gating is where the win lives (2026-08-15)

`transformer.verifyQmm`'s split-K and msg tiles read weights as
`w_q[(n0+j) * K_by_p + pack]` — one uint32, eight nibbles — so every non-4-bit
pack fell through to stock MLX on the whole M1-M4 line. Only the NAX m16 tile
(M5-class, `applegpu_g17`) handled 5/6-bit. The Qwen3.8-27B 6-bit pack is the
best quality-per-byte build for a 36 GB Mac (95.0% top-1 against bf16 vs the
4-bit pack's 83.6%) and paid the full lane-off penalty on every MTP round.

Both tiles are now `BITS`-templated with the same byte-addressed unpack the NAX
tile already validated: one `pack` iteration always covers 8 K values (that is
the Vec8 activation load), so its weights are exactly BITS bytes at
`w_q + n * K_bytes + pack * BITS` — 4 values per 3 bytes at 6 bits, 8 per 5
bytes at 5, one byte per value at 8. 4-bit keeps its original aligned uint32
read verbatim.

### Three things that were not obvious

**The kernel-cache key was a false alarm, and the reason is worth knowing.**
`getVerifyQmmKernel(m, bn)` is not keyed by bits, which looks exactly like the
`ShapeKey` collision class. It is not: MLX names its JIT'd custom kernel
`custom_kernel_<name>_<template_hash>_<input dtypes>_<output dtypes>`
(`backend/common/metal_kernel.cpp`), so template args ARE the cache key. That is
the same mechanism that already lets one cached `mlx_fast_metal_kernel` handle
serve GS 32/64/128. `BITS` rides in as a template arg and needs no key change —
and the parity test drives every width through one cached handle in one process,
which is the test that would have caught it.

**Hoisting the POINTER is not hoisting the LOAD.** The tile's shape is "issue
every column's weight word, then run one sequential chain per column", and the
first cut hoisted only `const device uchar* wp{j}` — leaving the actual byte
reads inside each chain, serialized behind the previous column's arithmetic.
The widened arms hoist the WORDS, at the widest load each width's alignment
allows: 8-bit takes two uints (`pack * 8`; not one ulong — 4-byte alignment is
all the shipped 4-bit path assumes of this buffer), 6-bit three ushorts
(`pack * 6` is only 2-aligned) re-assembled into two 3-byte words, 5-bit bytes.

**Shape-gating is most of the result, not a detail.** Measured on an M4 Max
(no NAX lane, so the plain-SIMD tiles are all this machine has), Qwen3.8-27B,
MTP on, decode tok/s from the server's own `timings`, per-prompt medians over
two passes with the CELL order reversed on the second:

| arm | code | prose | agentish |
|---|---|---|---|
| 6-bit, every shape forced on (`MIXED_PLAIN=1`) | 1.03x | 1.05x | 1.03x |
| 6-bit, `N >= 5120` (`mixedPlainShapeEnabled`) | **1.10x** | **1.22x** | **1.17x** |
| 8-bit, `N >= 5120` | 1.03x | 1.04x | 1.01x |

Adopting the 1024-wide K/V projections gives back ~10 of the ~20 points — the
same split `mixedNaxShapeEnabled` encodes for the NAX tile, and the reason
adoption is a measured predicate rather than a width list. For scale: killing
the 4-bit lane outright on this box costs 1.09-1.12x, so the 6-bit lane is worth
MORE than the 4-bit one it was modelled on.

8-bit ships default OFF at 1.01-1.04x — inside what the harness resolves, and a
lane that only adds dispatch surface for that is not worth defaulting. 5-bit has
the kernel arm and the parity test but no pack on the measuring box, so it does
not get to ride 6-bit's number either. `MLX_SERVE_VERIFY_QMM_MIXED_PLAIN=0|1`
forces both arms.

**Cell ORDER is a real bias in a paired sweep.** The first cell of a pass reads
~2% fast; a control pack that ignores the lever entirely measured a 1.8-2.8%
"gain" between two cells that differ in nothing. Reversing the cell order on the
second pass (not just the pack order) cancels it — before that fix the 6-bit win
read as +1 to +5%.

### The end-to-end bar the plan asked for is not achievable, and the control says so

"Greedy temp-0 continuation must be byte-identical with the lane on and off" is
not a property this lane family has. The verify tile accumulates the fp32
reduction in a different order than stock qmm, so near-tie argmaxes flip — the
documented INT4-divergence class. The SHIPPED, unchanged 4-bit lane diverges
from stock under exactly that test and diverges EARLIER (char 322 against the
6-bit lane's 557), both continuations fluent and on-topic.

The bars that do hold, and that this change is pinned against:
- parity against an fp32 DEQUANT reference at real shapes, per width, never
  kernel-vs-kernel (`verifyQmm: the plain-SIMD tiles match stock qmm at
  5/6/8-bit affine too`, red-verified by flipping the 6-bit mask to 0x1F and by
  reversing the 8-bit byte order);
- **4-bit bit-identity across the refactor**: same binary except the kernel
  generator, same 4-bit pack, MTP engaged in both arms, 250 greedy tokens
  identical to the byte.

## Fused kv-quant reads under spec decode: the verify widths were the whole bill (2026-08-15)

**Symptom.** With `--kv-quant 8` (or 4) and `--kv-attn-mode auto`, decode on MTP
models collapsed past 8k context: 28 tok/s where the dense read does 60
(Qwen3.6-27B-oQ4e and Qwen3.8-27B, M4 Max). It read as a "long context
regression" in llmprobe. It was NOT a regression — byte-identical behavior on
fc6b5ce, 5d10cb7 and v26.8.6; the old llmprobe cards it was compared against
had run kv-quant OFF. The repro needs all three legs: kv-quant on, ≥8K prompt,
PREDICTABLE content (deep MTP rounds). Random filler shows no cliff because
acceptance collapses and rounds stay cheap. Harness kept at
`~/claude-tmp/kv8-cliff-20260815/`.

**Mechanism.** `kvAttnFusedEligible` admitted `t_q <= 32` once the context
passed the auto crossover. Inside the fused arm, decode width (t_q == 1) rides
`qkvAttnDecodeKernel` — the measured +10..+56% serial win — but verify widths
(T_q 2..7 under MTP) fell to `kv_quant.quantAttention`: a composed chain of
quantized matmuls against the PACKED K and V, per layer, per spec round. qmm at
M 2..7 over an 8k+ contraction is the exact dead zone the verifyQmm weight
lanes exist for, except here it ran 16 attention layers x 2 packed matmuls
every round. Round time measured 63 ms at 4k (below the crossover, dense path)
vs 158 ms at 8k (composed packed path), flat to 16k. Under MTP nearly every
trunk forward IS a verify, so the fused win case barely ran. The composed
verify chain was a deliberate choice that predated deep-MTP defaults and was
never A/B'd at spec depth.

**Fix.** `kvAttnFusedEligible` requires `t_q == 1`. Verify and small-prefill
widths fall through to the dense dequant + SDPA path — the reference semantics
anyway. Decode-width eligibility is unchanged, so the serial fused win is
untouched by construction; bits-agnostic, so kv4 and kv8 are both covered. Side
effect, documented: a forced per-request `kv_attn_mode:"fused"` at verify width
now gets the dense read too — honest, since the composed chain it used to get
is the slow path this fix removes. The lazy sliding sel_mask null guard inside
the fused arm stays (only t_q == 1 reaches it now, but the live failure it
fixed was real and it costs nothing).

**Guards.** `kvAttnFusedEligible` unit test (red-verified: pre-fix admits
t_q == 2) + `tests/test_kv_quant_fused_equivalence.sh`, which gained a spec-on
arm: PLD engagement (`[spec-stats] mode=pld`), fused engagement with the
one-shot `[kv-attn]` line's `Tq=1` (no other width is eligible now), and greedy
first-N equivalence vs the dense spec arm. The decode-rate recovery is NOT a
shell-test assertion — a one-run spec cell is variance per /bench rules; it is
pinned by same-boot interleaved perf A/Bs.

**What Phase 2 would be (not built).** The dense fallback still pays a full-KV
dequant transient per verify round. A real packed verify-attention kernel for
M 2..8 (the verifyQmm move applied to attention) is the prize IF the dense-arm
rate ever matters vs serial. Bars if attempted: fp32-dequant-reference parity
per width, no-worse-than-dense same-boot A/B at 8k/16k/32k per arch,
engagement log per width, kill switch, per-shape eligibility A/B. Non-goals,
decided: the 8192 auto crossover stays (fine for the path that remains); no
resident dense-KV cache to amortize verify dequants (it re-spends the memory
kv-quant exists to save, exactly on the machines that need kv-quant).

### Phase 2 follow-through: the verify widths got their own packed kernel, and two pre-existing losses surfaced on the way (2026-08-15)

Sizing the prize before building (dense-fallback vs kv-off, spec-on,
Qwen3.6-27B-oQ4e-mtp, same boot): −2% @ 8k, −7% @ 16k, −12.6% @ 32k — the
full-KV dequant transient per verify round grows with context, so the plan's
"only if it matters" bar was met.

**`qkvAttnVerifyKernel`** (T_q 2..8, `MLX_SERVE_KV_ATTN_VERIFY=0` kills):
QKV_DEC's two-phase block-parallel layout generalized to TQ query rows per
head. q is read from DEVICE, not staged (LQ=GQA·TQ rows of DK f32 is up to
49 KB of tg memory; the reads are lockstep-coalesced and L1-scale); the
causal TAIL is applied from geometry (`gpos < Tk - TQ + 1 + tq`, the same
end-aligned contract as SDPA "causal", which is what the dense path serves
these widths — mutation-checked: dropping it fails parity at max_err 0.076);
"array" (sliding) masks decline; scale folds at score write. Partials merge
through the SAME `qkvMergePartials` carry (extracted from the decode wrapper,
pure code motion), rows (head-major, tq inner) so [Hq·TQ, DV] reshapes
straight to [1, Hq, TQ, DV]. Config cache is one slot per TQ — verify widths
alternate round to round, a single-entry cache would rebuild per call.

Measured (M4 Max, kv8, spec-on, same-boot per-request interleave, medians):
qwen3.6-27B auto 60.7/49.1/40.5 vs dense 58.4/46.1/38.4 vs kv-off
60.9/49.3/41.9 at 8k/16k/32k; qwen3.8-27B 66.0 vs 64.3 vs 67.2 at 8k. The
kernel recovers most of the dense-fallback tax — kv8 now decodes ≈ kv-off at
every rung with HALF the KV bytes. Engaged at Tq 5-7 live (auto-depth MTP).

**Two pre-existing t_q==1 losses found by the gemma4 sanity pass** (auto was
a 1.45x decode LOSS on gemma4-12B kv8 at 11k — 21.1 vs 30.5 tok/s, present
before Phase 1):
- gemma4's full-attention layers (gqa 16 × dk 512) overflow the decode
  kernel's threadgroup q-staging budget, so EVERY decode step fell to the
  composed chain: −17% alone. Fix: the fused arms are kernel-or-DENSE now;
  `kv_quant.quantAttention` never serves (scan-pinned to its one parity
  caller). A declined kernel falls through to the dense arms.
- gemma4's window-1024 sliding layers sat exactly AT the 1024 per-layer kv
  floor and engaged the masked kernel at ~1k KV: another ~4 tok/s. Fix:
  `KV_ATTN_FUSED_MIN_TK` 1024 → 2048 (Laguna's 512-window calibration point
  stays excluded either way).
After both: gemma4 auto == dense (30.4/30.4), zero engagements — correct,
its shapes are outside every measured adoption set. The verify kernel's
gqa16×dk512 cell measured −1% at Tq 2, so the wrapper gates dk ≤ 256 — the
mixedNaxShapeEnabled rule again: adoption is what was measured, per shape.

Guards: `qkvVerParityCase` (TQ 2..8 × bits × GQA incl. the 24/4/256 live
geometry, strided views, multi-block merge, vs dense SDPA over the SAME
dequantized view), the decline test, the kernel-or-DENSE source scan, the
verify-eligibility unit arms, and the shell script's third arm (qwen-shaped
model, PLD on): verify-kernel ENGAGEMENT is the assertion — the dense
fallback is output-equivalent, so a dispatch hole is invisible to equality.
gemma-4-e4b deliberately has NO verify-engagement assertion: its shapes are
outside the adoption set, and asserting one would be a checkpoint
expectation.
