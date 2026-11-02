# Engine: KV cache, spec decode, kernels, MLX: gotchas

One story per gotcha: the defect, the cause, the fix, the guard. The rules live in the root `AGENTS.md` (`## Rules`)
and each plugin's `AGENTS.md`; what is still open lives in `docs/known-issues.md`; spec-decode and cache designs live
in `docs/reference.md`. An env lever a story names that `src/env.zig` no longer lists was retired once its arm was
decided: the measured default stays, the losing opt-in was deleted. Engine code is under `src/engines/mlx/`
(`spec/`, `cache/`, `kv/`, `sampling/`, `generate/`), shared blocks and kernels under `src/sdk/blocks/` and
`src/kernels/`.

## Sampling

### `top_p: 0` masked every token and sampled uniform garbage
Defect: a literal `top_p: 0` (what clients send for greedy) returned noise over 248k ids. Cause: the nucleus keeps a
rank while the mass strictly above it is `< top_p`; rank 0 sees 0, so nothing was kept. Fix: the threshold floors at
`floatMin(f32)`: `top_p 0` is greedy. Guard: `applyTopP at top_p 0 keeps exactly the argmax`,
`tests/test_api_edges.sh`.

### A penalty applied only on the logprobs path was ignored everywhere else (#564)
Defect: "apple" x40 at `repeat_penalty 2.0` gave 120 apples, and 3 with `logprobs: true`. Cause: penalties reached
only `sampleToken`; the fast path, grammar, spec verify and batched decode sampled raw logits. Fix:
`Generator.sampleLazy` penalizes over the realized ids; `penalized()` keeps a request off the fast path, spec and
batching. Guard: `repeat penalty shapes every sampling path`, `tests/test_repeat_penalty.sh`.

### OpenAI's `frequency_penalty` was read as a `repeat_penalty` multiplier (#912)
Defect: Crush's `frequency_penalty: 1` halved every logit of every token already generated; once #564 applied it
everywhere, Qwen3.8 wrote tool calls as prose. Fix: `frequency_penalty` is its own field, subtracted per occurrence,
as vLLM and llama.cpp read it. Guard: the frequency-penalty tests in server and generate.

### Every sampled token ranked the whole vocabulary
Defect: sampled decode was 7% slower than greedy on Flash Next: `applyTopP` argsorted 248k logits per token (on Metal
partition and argpartition ARE the full merge sort). Fix: `topRanksDescending` takes the exact top m via chunk maxima;
the shortlist breaks ties exactly as the whole row does (stable ascending argsort of the negated row, lowest chunk ids
among equal maxima, chunks sorted by id); the nucleus scan runs in f32 (mlx's cumsum at bf16 dropped columns). Guard:
shortlist == full-row route byte for byte over many settings.

## KV cache and prefix reuse

### Hot-cache restore must clamp the offset to the matched length
Defect: gemma-4-26B at ~16K crashed with `[broadcast_shapes] (1,1,Q,16890) and (1,H,Q,16892)`. Cause: PLD left stale
draft rows past the committed step, the commit snapshotted the inflated buffer, and a partial hit skipped the
truncate; RoPE offset and buffer length drifted. Fix: always `truncate(final_len)` on restore. Tell: two shapes
differing only in the key dim after a `[hot-cache] reused N/M` where N is the entry's full length. Guard: `restore
clamps an inflated snapshot to the matched length`.

### KV invalidation after tool calls and pad-only generations
Defect: reusing a cache whose generated tool-call tokens are not in `cached_prompt_ids` corrupts attention. Fix:
invalidate after tool calls and pad-only generations.

### Sliding layers keep the full buffer; the view is what is trimmed
Defect: (design) Gemma 4 E4B's 512-token window. Fix: the buffer keeps every row, as mlx-lm's `RotatingKVCache` does;
`slidingViewFor` sizes the view the forward reaches (`window + q_len - 1`), at prefill too (it was once decode-only,
see below).

### A short restore kept the long donor's KV capacity (#492)
Defect: a "hi" chat matching only an 80k session's system prompt kept 80k of capacity (1.9 GB), and the byte budget
then evicted the long session. Cause: a restored cache shares the donor's buffer and the first append copied all of
it. Fix: `KVCache.restoredOversized` regrows from the prefix when the donor is much larger; evict-to-admit is no
longer qwen4-only. Guard: `KVCache: an append after a short restore sizes its copy from the prefix`.

### A warm turn that could not share its prefix was refused (PR #518)
Defect: a 46k-of-47.7k-warm agent turn on a 64 GB Mac got `PrefillDoesNotFit`. Cause: off SSD-first a restore shares
buffers and the first append copies the whole prefix; the bill was honest. Fix: a full-entry hit that cannot share
takes the checkout on demand (`checkoutRestored`) and re-bills; a repeated one-token prompt that fully matches its own
entry is declined (it left nothing to forward and segfaulted). Guard: the move-on-demand and one-token tests.

### A disk restore was billed as a cold prompt (PR #621)
Defect: 78k of 78.5k tokens persisted and the turn refused by 1.2 GB; a 128k turn refused by 93 MB. Cause: the disk
arm left `checked_out` false, so `WarmPrefix` credited 0. Fix: a disk restore sets `slot_owned` and is credited.
Guard: `slot_owned` assertions in the restart and hybrid-disk tests.

### Disabled prefix retention reserves nothing
Defect: `--prefix-cache-entries 0` still reserved the RAM budget in context and chunk sizing. Fix: a disabled cache
reserves zero bytes; a byte cap is not proof a cache exists. Guard: the sizing tests.

### Quantized KV (`--kv-quant 4|8`): contract and traps
Defect: f16 packs (Bonsai 2) under `--kv-quant 8` decoded 2.6x slower: scale buffers were hardcoded bf16, so the first
attention layer widened the residual stream to f32. Fix: the cache hands attention the dtype it was fed. Contract:
attention reads `KVCache.denseView` on every path; a scheme the fused kernels cannot read pays a whole-cache dequant
per token, which is what killed TurboQuant (removed, 14-33% slower). 4-bit KV stacks with INT4 weights in long-greedy
divergence: byte-stable long greedy needs `--kv-quant 8` or off. Guard: `KVCache affine quant hands attention the
dtype it was fed`, `tests/test_kv_quant_equivalence.sh`.

### The hybrid prefix cache retains far more than it reports
Defect: on Qwen3.5-4B (8 attention + 24 GDN layers) active memory climbed ~2.2 GB per agent turn and never released.
Cause: each commit snapshots every linear layer's conv/SSM state and the arrays refcount-share their prefill-chunk
parents, so retention was ~3.4x the reported bytes and the byte cap never evicted enough; `--kv-quant` does not touch
SSM state; the app silently launched 32 entries. Fix: the entry count is the lever (`ramCappedPrefixCacheEntries`),
always emitted by the app; snapshots are materialized as owned copies (`materializedOwnedCopy`, below).

### A slice that must not alias needs an arithmetic copy
Defect: "materializing" a slice still retained its parent buffer (the hybrid cache under-count). Cause: `mlx_copy`
shares the input buffer and `mlx_contiguous` no-ops when strides read as row-contiguous. Fix: `materializedOwnedCopy`
(add a same-dtype zero: a real kernel with a fresh output) + eval for any slice that outlives its parent. Trap:
`mlx_save_safetensors` appends `.safetensors` when the path lacks it. Guard: `captureSsmCheckpoint materializes state
copies`.

### A slice handle wrapped in `contiguous` and never freed pins its parent
Defect: MageFlow leaked ~2.2 GB per megapixel per generation, surviving unload, while RSS sat flat. Cause: six helpers
wrapped a `mlx_slice` view in `contig` and never freed the view, which pins its parent. FLUX klein was the clean
control; forcing an eval per block still grew memory, ruling out lazy retention. Fix: one `sliceContig` owns and frees
the view; same PNG bytes. `mlx_clear_cache` is not the fix: if `active_bytes` climbs, you hold handles. Guard:
`tests/test_media_gen_memory.sh` (varies the size), `materializing helpers hand back every array they take` (the
source rebuilt per iteration: a caller-owned source hides the leak).

### A scope freed after the eval holds every activation of the step (Stable Audio 3)
Defect: a 120 s clip peaked at 45 GB on a 2.6 GB model with perfect parity. Cause: the step's `Scope` was
deinitialized after `mlx_array_eval`, so every handle pinned its buffer through it. Fix: close the scope before the
eval and keep only the output: 4.2 GB, and faster. Guard: `sa3: a 30 s sample + decode holds no more than one step's
working set`.

### Lazy `mlx_slice` views fed to `gather_qmm` compute wrong expert outputs
Defect: a split `gate_up_proj` produced exactly-zero MoE outputs at real-checkpoint scale while toy shapes and
data-pointer reads were fine. Fix: any weight born from a slice that feeds `gather_qmm`/`quantized_matmul` is
`mlx_contiguous`-materialized at load (`splitPackedGateUp`); assertions must go through the kernel, not a data read.

### mlx-c iterator refs: `iterator_next` hands you a +1
Defect: every video load-generate-unload cycle leaked the whole engine (~18 GB; RSS reached 100 GB), presenting as an
unload bug. Cause: `ltx.loadComponent` copied the iterator's value with `mlx_array_set` and dropped the original, a
phantom +1 on every tensor. Fix: transfer the handle or free it. Tell: `/props` `active_bytes` not dropping after an
unload means leaked handles. Guard: `loadComponent releases every tensor on deinit`.

### A loop whose buffer sizes never repeat must release the allocator cache
Defect: a 14 GB model reached a 92 GB footprint across requests with growing prompts, and a 36k prompt peaked at 96 GB
inside one request, with `active_bytes` flat. Cause: freed buffers return to MLX's cache, which only serves the same
size later; chunked prefills free a different size every chunk. Fix: `mlx_clear_cache()` once per prefill chunk and
once per emitted block, un-skippable. The repro must vary the shape that drives allocation size, and sample the
footprint during the request (`vmmap -summary`, `Physical footprint (peak)`).

### SSM state init, the parameter-free norm, Nemotron's dt limit
Defect: three load-time traps on hybrid trunks. Fix: init checks `ssm_state.ctx == null` (`initialized` is set after
the conv update, before the recurrence exists); a parameter-free RMS norm passes `ones([dim])` (mlx-c crashes on a
null weight); Nemotron-H clips dt only by `time_step_limit` (numeric or positive infinity), never `time_step_min/max`.

### PLD on hybrid SSM: snapshot fields are populated independently
Defect: PLD on LFM2 aborted the process in `mlx_array_set` with a null source. Cause: gated-conv layers write
`conv_state` only and set `initialized`. Fix: every snapshot and restore checks each field's `.ctx` independently.

### A special-token scan of O(specials x text)
Defect: gemma-3 (6,415 specials) spent ~12 s of CPU tokenizing a 66 KB prompt on every cache miss; the bug shipped
twice (tokenizer and a chat-path duplicate). Fix: one splitter, `Tokenizer.encode`, a single pass over
first-byte-bucketed candidates (longest first). Any per-position loop over a vocab-derived collection needs an index;
`timings.tokenize_ms` and `/tokenize` isolate the tokenizer. Guard: the special-token scan characterization test.

## MLX build and FFI

### The Homebrew mlx bottle ships with NAX silently disabled
Defect: stock ops never dispatched M5 neural-accelerator kernels. Cause: NAX is gated at CMake configure time on macOS
SDK and deployment target ≥ 26.2; brew's builders ran 26.0 and the gate fails silently. Fix: mlx + mlx-c are pinned
submodules built by `scripts/build-mlx.sh` at deployment target 26.2; the script and `tests/test_mlx_staged_nax.sh`
assert `*_nax` in the metallib. Consequences: macOS 26.2 minimum; M1-M4 behaviour unchanged; new same-methodology
bench baselines.

### An mlx-c pin must be a pair that compiles against the pinned mlx
Defect: the mlx-c v0.6.0 tag does not compile against mlx 0.32; mlx 0.32.2 added `force_fused` to sdpa. Fix: pin the
mlx-c commit that builds (a PR head fetched by full SHA), re-diff the headers against `src/mlx/ffi.zig`, run the
suite. `force_fused` on a shape without a fused kernel throws, and hd 512 (Gemma 4's full layers) has none: gate it
(`naxSdpaPreferred`, `sdpaForceFusedFor`). The fused kernels return a strided view: read results only after
`mlx_contiguous`.

### A non-absolute path to `openDirAbsolute` miscompiles the caller
Defect: a headless boot fell into `parseConfig("")`: a branch visible in the source was absent from the ReleaseFast
binary. Cause: `openDirAbsolute("")` asserts, the assert is `unreachable`, and the UB let the optimizer delete a later
branch. Fix: guard every absolute-path open against empty or relative input. When ReleaseFast and Debug disagree on
control flow, build Debug to locate the UB.

### `mlx_array_new_data` copies shape-worth of bytes
Defect: a guard-page test read past a buffer. Cause: `mlx_array_new_data` copies as many bytes as the shape implies
from the pointer it is given. Fix: size the source to the shape.

## Speculative decoding

### Verify invariant, and the off-by-one that corrupts silently
Defect: (design) every drafter shares one verify contract. Fix: `cache.step = prompt_len + emitted`; t1 not in cache
on entry; verify input `[t1, draft…]`; a partial accept's correction is sampled from the ORIGINAL
`verify_logits[accepted]` (index `accepted`, not `accepted-1`). Guard: `tests/test_pld_equivalence.sh`.

### A speculative token cap belongs before state commit
Defect: near a 17-token cap, PLD returned a five-token block whose state and usage were committed while the client got
one token; DSpark, MTP, the Gemma drafter and PLD all shared the shape. Fix: every path caps the accepted prefix
(`capAcceptedForTokenBudget`) before committing any representation of the round (DSpark rolls its module back at the
capped boundary); one publisher increments once per pushed token. The DFlash break-even threshold scales with the real
draft width; a DFlash payload commits only when the context length matches the trunk. Guard: lint pin + the
block-publisher tests.

### No KV snapshot across the verify forward
Defect: at 64k context each MTP round copied ~4.3 GB: a live snapshot refcount-shares the buffers, so the verify's
`slice_update` cannot donate and copies per layer per round. Tell: round time grows with context far faster than the
verify forward. Fix: rollback is scalar anchors + per-position SSM capture + offset-only `truncate`; snapshots only
where a real restore-and-re-forward happens.

### A spec-verify forward is decode-shaped, not prefill-shaped
Defect: MTP rounds on the 64-layer 27B paid 16 pipeline drains each. Cause: the prefill eval cadence keyed on `seq_len
> 1`. Fix: `prefillEvalCadenceApplies` (seq ≥ 32); a verify, canvas re-encode or batched head forward is not a
prefill. Guard: its unit tests.

### PLD/drafter long-greedy byte divergence at INT4
Defect: speculative greedy output diverged from plain after ~30-80 tokens. Cause: verify runs qmm where serial runs
qmv, so near-tie argmaxes flip on INT4. Fix: none needed: both are valid greedy outputs, and sampled requests are
distribution-exact. Byte-stable long greedy means no spec (and `--kv-quant off/8`).

### The MTP head: committed history, quant per weight, norms that need folding
Defect: (port traps) sidecar heads mixed 5/6-bit q/k/v with 4-bit o and experts; Qwen and OptiQ store head norms
delta-encoded (the layer computes `1 + w`), and an unfolded head engages at ~0% acceptance while passing equivalence
tests; OptiQ quantizes `embed_tokens` to 8-bit beside a 4-bit trunk and reading it as 4-bit crashed the server. Fix:
quant params solved per weight from geometry; norms auto-fold when delta-encoded (`mtpNormNeedsRepair` reads the
negative fraction); embed params resolved per weight; verification always uses the trunk head; a pending history stash
makes `mc.step` stale (`mtpRoundOff0`). Guard: the detect/fold and optiq tests in `sdk/spec/mtp/mtp.zig`,
`tests/test_mtp_equivalence.sh`.

### MTP auto-depth profiles are full-round tensor fingerprints
Defect: the Qwen3.8-27B 4-bit pack stayed at auto depth 6 though depth 8 ran 1.21x faster on the G17 NAX lane;
treating sidecar q4/gs64 as sufficient would have mispriced oQ4e's mixed trunk. Fix: a calibrated `MtpCostProfile` is
chosen from the whole runtime surface (trunk projection layout, embedding storage, every sidecar projection, draft
head, live NAX), collapsed into one enum; unknown combinations stay generic at cap 6; profiles are revoked when their
NAX lane is unavailable. Guard: the profile table in the format corpus, `MTP_EXPECT_AUTO_PROFILE` in
`tests/test_mtp_equivalence.sh`.

### Verify-width split-K qmm (the MTPLX port)
Defect: M 2..8-row verify shapes sit in stock qmm's dead zone. Fix: `verifyQmm` routes eligible affine qmms through
split-K (M 2..7), a wide tile for lm_head-class N, and the NAX m16 tile on G17 (M 8..16; mixed widths shape-gated per
width after a sweep). Rules: codegen named scalars with literal accumulator indices (an indexed array spills 10x); M =
8 is the plain-SIMD cliff; the NAX kernel is never built where the probe is false; a templated width `else` is not a
supported-width list (q8 compiled through the q6 arm and returned garbage: each width needs its own arm and parity
row); A/B each adopted shape on its own model (the lane lost on the Gemma drafter's M = 5); rebuild the exe before a
live A/B. Parity: no-worse-than fp32 dequant truth, never kernel-vs-kernel (a kernel-vs-kernel bound went red on the
first CI GPU). Guard: `VerifyQmmParity`, the lane table test.

### DFlash outranks the MTP head, and one place derives the spec mode
Defect: a pack shipping both an in-checkpoint MTP head and a `drafter/` never ran the drafter. Fix: a loaded DFlash
sidecar wins (`--no-drafter` hands rounds back to MTP); eight hand-rolled server sites read one `requestSpecModes`,
twinned by `specInitWiring` + `specTickMode`.

### Wide verify pays at four lanes, not at one
Defect: four concurrent requests ran 71.9 tok/s against Inco Splash's 94.6 on an M4 Max 27B. Cause: no verify lane
past 7 rows off-NAX; lanes planning solo depths wasted a group's rows; the pad-waste cap read `cache.step` (0 on a GDN
trunk) and the stacked arm copied every slot's KV past 1024 tokens. Fix: the `matmul2d` tile runs as shader code on M4
(8/16/24-row tiles); 4..8 lanes share one 16-row verify (`mtpSubGroupPlan`); attention per slot past 1024
(`perSlotBatchedAttn`). N=4 71.9 → 109; then group lanes draft as rows of one head forward (`mtp.forwardLanes`) with
the next chain pre-dispatched: 109 → 119.5. Guard: `per-slot batched attention equals the stacked arm`, `mtp:
forwardLanes equals N solo steps`, `tests/test_mtp_batched.sh`.

### Auto MTP depth: plan tokens from the acceptance EMAs, price time from the table
Defect: auto read 3-4% under forced depth 3 on code; base depth wandered 2..5. Cause: the chunk-A confidence read
blocked the CPU; the planner scored widths with the table's `tok` column, which only the standing width refreshes.
Live A/Bs could not decide (greedy text diverges with depth). Fix: `MtpDepthPolicy.accept` below 8192 KV tokens:
tokens from the EMAs, time from the table, one chunk, a width trial as the probe; above 8k the legacy planner is
cheaper. Instrument: `engines/mlx/spec/mtp_replay_test.zig` over recorded acceptance traces
(`tests/record_mtp_accept_traces.py`), with a measured chain-degradation table; it reproduces the live ladder within
0.4%.

### Wide verify does not pay on an M4 Max at N=1
Defect: (measured null) wider blocks raise acceptance and lose throughput. Measured: DFlash2 b5 peaks (82.0 code /
50.4 prose); b8 clears 100 tok/s predictable and drops novel below serial; the M4 block cap stays 5,
`wideVerifyLaneAvailable` stays NAX-only, `WidthChooser` stays opt-in; qmm is 69% of the verify marginal at 11.1
TFLOP/s. Lifting the fused GDN row cap from 9 changed output bytes (the fused `beta` differs from `mlx_sigmoid` by one
bf16 ulp on rare inputs), so "bit-identical at S 1..9" there is a sampling result, not a proof.

### A sampled draft proposal must use the shortlist the rerank built
Defect: at temperature 1 on Flash Next the greedy proposal accepted 0.50/0.22/0.09 per index on prose; the sampled
proposal lifted acceptance 7-12 points and netted zero because it dropped the coarse shortlist for the full 675 MB
head. Fix: `shortlistProposal` samples from the rerank's 32 candidates through the shared sampler (q is zero off the
shortlist, exact under Leviathan); the proposal is chosen per request (`mtpDraftGreedyFor`); group rows use one
filtered block. The Lightning-style sharpened stochastic draft loses to greedy on low-entropy content. The draft
temperature is per family.

### MTP round pipelining and the qwen4 lazy pre-draft
Defect: (levers) the CPU graph build was the recoverable overhead of an MTP round. Fix: dispatch the draft chain as
soon as it is built; on qwen4 a solo greedy round builds the next chain from its lazy verdict through a padded head
(dead rows masked, `HeadPlace`), discarded on a cut; the plan is drawn one round early by design (an auto depth change
lands a round later), so the byte bar is a pinned depth. Measured +2.6% greedy on M5 Ultra; host encode bounds what
overlap can buy.

### The EV controller under honest costs
Defect: two-chunk plans paid a mid-pipeline sync the cost surface could not see, and an EMA dragged cold closed the
extension horizon forever. Fix: an extension dry-spell gate fed by realized extensions; one extension position stays
reachable while the base pays. Refit costs only on saturated sweeps whose realized `m_avg` equals the depth; thermal
soak lies (same-session ratios only).

## Allocator, dtypes, kernel arguments

### A kernel's dtype comes from its inputs (GDN blocked prefill on an f16 pack)
Defect: one request on an f16 Bonsai pack killed the server: `Unable to build metal library … const device bfloat * …
const device float *`. Cause: the blocked GDN kernel hardcoded `InT = bfloat16` at five sites; f16 weights promoted
q/k/v to f32 while `g` and the state stayed bf16; the comment had predicted it and dismissed it. Fix: `InT`/`StT` from
each input's actual dtype, the store type from the output (`OutT`), and the block size follows the dtype
(`gdnBlockTFor`: f32 staging doubles threadgroup memory, so TB 16, or decline). Read the dtypes out of the mangled
kernel name in the error first. Guard: `gdnBlockTFor`, f32 and f16 cases in the blocked-parity sweep.

### MLX's buffer pool is unbounded, the clear cadence was skippable, and KV growth orphaned a cache every 256 tokens (#110)
Defect: a 27B MTP session in Zed reached 81.4 GB of footprint against 19.6 GB of GPU memory. Cause: MLX's pool trims
near 91-121 GB on a 128 GB Mac; the drafter, MTP and batched paths never cleared it, and `step % 256` can step over
every multiple at a variable spec stride; linear 256-token KV growth stranded ~5.6 GB per event at 89K context (MLX
reuses a buffer only within two pages of its size). Fix: `applyMlxCacheLimit` once in `main()` (`max(2 GB, min(8 GB,
RAM/16))`); `Generator.advanceStep` is the one step mover and clears on an interval; `finishSlot` clears; KV grows
+25% (floor one chunk, cap 8192). 1544 → 274 MB pool in the hermetic test; bench ABBA decode -0.0%. `/props`
`memory.cache_bytes` and the tray now show the pool. Guard: `KV growth does not ratchet the MLX buffer pool`, `clear
cadence survives variable spec strides`, `tests/test_kv_cache_growth_memory.sh`.

### A load-time f32 constant table made Laguna 3x slower
Defect: Laguna XS decoded at 40.5 ms/token, 2.36x slower than the mlxfast-challenge tree's unoptimized baseline on the
same silicon. Cause: the YaRN mscale table was built f32 and multiplied into bf16 q/k, so the whole residual stream
ran f32 from the first full-attention layer and every later weight read was upcast; op count, kernels, KV and host
syncs were all innocent. Found by a rung ladder (`diagProjBench`) where the reconstruction diverged from the shipped
function. Fix: `constTableAs` returns the table in the activation dtype (also the stock MLX path's numerics). 40.5 →
13.4 ms/token, within 2.2% of the frontier. Measure the reference tree interleaved in the same window: a mid-session
1.62x machine slowdown hit both alike. Ruled out by live A/B: command-buffer batching, wired limit, the pool cap,
compiled forward, an intra-step async-eval ladder (worse at every stride: we already overlap at the step boundary).
Guard: `constTableAs hands a load-time f32 table back in the activation dtype`.

### Per-channel f16 tables turned an f16 pack's residual f32
Defect: Bonsai 27B ran 14.8% slow. Cause: the checkpoint is f16 (its config says bf16; read the checkpoint), and 1-D
f16 norm weights promoted a bf16 residual at layer 0. Fix: `narrowsLoadedF16` narrows 1-D f16 tables at load (matmul
operands untouched: that is a kernel-selection change). Every forward path now arms a `[dtype-trace]` that logs only
where the dtype moves (lint pins). Measurement trap: the first boot after another large load and thermal soak each
reversed the sign; only paired A/Bs with a cooldown were stable.

### What a 3x forward fix invalidated: two defaults and a spec gate
Defect: decisions calibrated on the slow Laguna forward were wrong on the fixed one. Measured: the f32 KV half was
real (exactly 2.00x on the KV term) but the blended speedup falls with context (3.00x → 2.61x at 64K): terms improved
by different factors do not compound; `gatherQmv` now lost to stock `gather_qmm` on both Laguna sizes, so no arch opts
in by default; the PLD yield-gate warmup dropped 32 → 8 (recovers the losses, keeps the +77% echo win; enabling
speculation costs ~1% even when the prompt gate declines). Method: per-cell medians over repeated runs (single runs
lied three times); check a flagged cell's reachability from the change before re-running; diag `decode-profile`
destroys pipelining and is not a sizing tool; a ladder guard must use the ladder's own geometry.

### Fusing decode dispatches: only the critical path pays
Defect: a fused attention-output gate (four dispatches into one, bit-identical) measured zero or worse, though a probe
priced a dispatch at ~1.6 µs. Cause: the gate depends only on the layer input and already overlaps the projections:
deleting overlapped work buys no wall clock; a first version also ran one threadgroup over a whole row. Fix: fusions
on the dependency chain shipped (MoE router, SwiGLU: -2.7%, byte-identical); off-chain ones were deleted. Price
dispatches with an injected multiply-by-1.0 that feeds the chain, never "duplicate and read the marginal" on anything
that mutates state.

### Reproducing an MLX op means its reduction tree and accumulator
Defect: a "same softmax" router kernel diverged from the chain at token ~80 on gemma4-26B. Fix: three corrections,
each found by asserting bit equality with the chain: `fast::exp` and a multiply by the reciprocal; MLX's
virtual-thread reduction tree (four consecutive elements, then simd sums); the top-K `sum` accumulates in the input
dtype (bf16), one thread per 8-wide row; MLX's two-sided sigmoid.

### A JIT kernel and MLX's metallib disagree on transcendentals
Defect: a fused SwiGLU with MLX's sigmoid source verbatim diverged live at token ~55; random-sampling tests said
bit-identical. Cause: `metal_kernel` JITs in Safe math mode, unlike the metallib; exactly one bf16 input of 65536
differs, and an 8192-wide MLP hits it within dozens of tokens. Fix: sweep all 65536 inputs, never sample; the kernel
indexes a table built once from `mlx_sigmoid` over every 16-bit pattern (`swigluSigTable`); f32 declines.
`mlx_compile` of the same chain is not output-preserving either.

### The fused gate+up turned a losing MoE kernel into the default
Defect: `gatherQmv` lost to stock and was demoted. Cause: it rebuilt its kernel config every call (+0.4 ms of graph
build per token, 60% of its deficit) and ran three serial dispatches. Fix: configs cached by geometry;
`gatherQmvGateUp` computes both dot products in one simdgroup and applies SwiGLU before writing: 2-3% faster; the
predicate is "would the fused kernel engage" (`useGatherQmvDecode`: silu, matching gate/up geometry), never a model
list. A cumulative figure comes from one window with both arms interleaved (a chained cross-window -5.8% was really
-4.2%).

### Fused residual + RMSNorm and fused QKV: two nulls
Defect: (measured nulls) `add` + `rms_norm` fused on the critical path lost 4 of 6 pairs; fused QKV (a weight-layout
concat) was null at decode and contradictory at prefill, and additive in memory. Cause: being on the critical path is
necessary, not sufficient: the fusion must remove real work (a 4 KB row is not); at prefill each projection is already
a compute-saturating GEMM. Traps: concatenating pre-transposed weight views changes which kernel runs (join on the
output axis and hand back a transposed view), mlx picks gemv kernels by shape so a layout fusion is not
output-preserving, and a reduction-order parity test needs a contraction dim near the real one (256 hid a bug 2048
showed). The fused add+norm later shipped in a different form (`add_norm.zig`, the MoE arm folding the K-sum).

### Decode-only dense-attention requant (`--decode-attn-quant`)
Defect: (lever) Laguna's bf16 attention was ~3 GB of a ~4 GB per-token weight read. Fix: an int8 g32 side copy built
lazily per weight, used at decode AND verify (never prefill), so draft, verify and serial score the same weights;
prefill stays dense and exact; tail layers past 80% use nvfp4-g16 (`attnDqFor`). -23% on Laguna XS, then -3.3% for the
tail; six-prompt characterizations diverged in wording only. Lossy, default on; the side copies are additive (~1.7
GB). A/B each newly adopted dense arch.

### Fused decode QK-norm + RoPE
Defect: (lever) the q/k chain before SDPA is serial, 4-6 dispatches per layer. Fix: `fusedQkNormRope`: cos/sin
extracted by probing the stock rope (never re-derived), every rounding boundary spelled out, bit-identical (max diff
0.0); the parity test caught `x1*sin + x2*cos` written symmetric (passes at offset 0 only). The eval-per-step µbench
read it +1.2% slower; the live A/B -1.6%: ship decisions come from the live number. Every qwen3.5/3.6 checkpoint is hd
256, so the qwen wiring is dormant: read the checkpoint before porting an hd-128 kernel.

### MoE down-path tail fusion and the certified lm_head prune
Defect: (levers) the decode expert tail ran down-gather, score multiply and K-sum as three dispatches; a certified
lm_head prune won a µbench. Fix: `gatherQmvDownReduce` shares the GEMV body text and replicates MLX's input-dtype
ascending sum: bit-identical, -1.5% live. The prune won 3 of 20 live pairs (and a later 5-bit readout repeated it) and
was deleted; a head readout is judged live, never by µbench.

### A cached `metal_kernel` config keyed on element count crashed the server
Defect: voice chat on Laguna XS killed the server after a few turns with `[matmul] … (1,16,512)` against
`(8192,2048)`. Cause: the fused SwiGLU's config cache keyed on element count; a 16-token restore suffix through a MoE
layer (16 x 512) and a decode through a dense layer (1 x 8192) collided, and the second got the first's output shape.
Found by replaying the debug log's request bodies interactively (a byte-exact replay at temp 0.7 did not reproduce)
and an env bisect. Fix: `ShapeKey` (full dims + rank) in every cached config that bakes a shape. Guard: `fused SwiGLU
config cache keys on the SHAPE, not the element count`.

### Quantized-KV fused decode reads: a no-op flag, NaN-blind parity, occupancy
Defect: `--kv-attn-mode fused` engaged on nothing (the live `update()` path returned no quant triples) and every
fused-vs-dense measurement compared dense with dense; once engaged, gemma answered `<pad><pad>` because `triu(ones) *
-inf` puts NaN below the diagonal, and every parity loop passed NaN (`NaN > x` is false); the kernel's first designs
lost by 5x and 1.7x. Fix: the equivalence script fails without the engagement lines; masks via `mlx_where`; parity
loops assert finiteness before diffing; the kernel keeps ≤ 10 KiB of threadgroup memory (256-row blocks) so several
groups stay resident: +10-56% at long context, with a per-layer floor where sliding layers regressed. Guard:
`tests/test_kv_quant_fused_equivalence.sh`.

## The SSM-checkpoint stride and hybrid prefix reuse

### The SSM-checkpoint stride was quietly the biggest prefill lever
Defect: the 27B prefilled 19-20% behind oMLX at 8K/32K: an 8238-token prompt ran in 33 chunks. Cause: the dense-hybrid
arm kept a 256-token checkpoint stride and every boundary ended a chunk, so every projection ran small-M qmm. Fix: the
stride never sub-divides the prefill chunk on any arch; SSM states are materialized in every chunk's eval batch (an
unevaluated end snapshot re-ran the whole last chunk's scan); the dense hd-256 chunk cap is 8192, MoE 4096; the
always-on snapshot sits `SSM_SNAPSHOT_BACKOFF` (30) tokens before the prompt end, because the generation-prompt suffix
renders differently once in history (64 pushed the tail into prefill cadence). Result: +10% at 2k, ties at 8k/32k,
which is the GEMM roofline.

### A changed image invalidates state from its media row, not token zero
Defect: a 135K Qwen3.8 vision-agent session ran ~9.5-minute cold prefills each time a new image appeared. Cause: exact
`vision_key` filtering rejected every entry before comparing tokens. Fix: entries record where media starts; a
differing image caps the match at the earliest media boundary (later superseded by per-item `MediaSpan` keys). Guard:
`tests/test_vision_prefix_cache.sh`.

### Hybrid lookup ranks the checkpoint it can restore
Defect: the same session alternated warm turns with cold ones at the image boundary. Cause: `findBestMatch` took the
longest raw token match, whose first SSM checkpoint sat past the match. Fix: a hybrid candidate scores its highest
checkpoint at or below its match (`findBestRestorableMatch`); a real four-turn run went from `0, 2048, 0, 2048` cached
tokens to `0, 2048, 2048, 2048`.

### A synthetic-dtype probe nearly shipped a 2x-bandwidth Inkling forward
Defect: a reference probe with my casts said the residual stream ran f32, implying f32 KV and experts. Cause: the
checkpoint stores the dense `global_scale` tensors as bf16; the probe proved the reference's semantics, not the
checkpoint's dtypes. The `[dtype-trace]` caught a widening at layer 2 (not 0): routing weights from the f32 router
multiplied bf16 expert outputs uncast. Fix: mirror the reference's `astype(x.dtype)`; dtype-gate raw data reads
(`mlx_array_data_float32` returns null on bf16, a silent no-op fold).

## DeepSeek-V4 (module-owned state)

### Module-owned decode state vs the prefix cache and teardown
Defect: DSV4 keeps all per-request state on the module; the prefix cache would attach, match, and set `cache.step`
without restoring it (serving the previous request's rings, or UB on a fresh boot), and unloading leaked ~108 GB. Fix:
such models are excluded from prefix reuse and owe a `Transformer.deinit` arm (now `ownsModuleDecodeState()`).

### A guard that shapes init options does not bind dispatch (the PLD bypass)
Defect: a DSV4 tool call came back with mangled DSML though two guards turned PLD off. Cause: the tick dispatched PLD
on the slot flag alone; `nextPld` trusted its caller; rejected drafts advanced module state past `cache.step`. The
`spec-gate:` log lines come from the connection thread and prove nothing about what the generator ran. Fix:
`Generator.pld_enabled` from the post-chokepoint options, `nextPld` self-declines permanently, dispatch through the
pure `specTickMode(slot flags x generator state)`. Regression trick: a prompt of every vocab id with `key_len=1`
guarantees a lookup match.

### Lazy stage weights + Metal's zero-filled OOM = fake 100% acceptance of `<BOS>`
Defect: DSpark answered two tokens then `<BOS>` forever with `avg_per_round=5.00`. Cause: warmup never touched the
DSpark stages, so ~11 GB of experts materialized mid-request past the working set; failing command buffers hand back
zero-filled buffers with no error for many evals, and zero draft logits agree with zero verify logits. All-zero +
independent chains + boot-to-boot nondeterminism is a memory signature. Also: `toHostF32` overread a strided f32 view
(same-dtype `astype` is a view; adding zero keeps a broadcast's strides). Fix: a reflective walker evals every stage
at load if `dsparkFitsBudget` on logical bytes, else DSpark is off by name; host reads check contiguity after eval and
materialize by flat reshape. Guard: `toHostF32 reads non-contiguous f32 views in logical order`, the budget test, the
full-accept seam test.

### DSpark round cost: the barrier, not the transfer
Defect: DSpark ran 13.5 tok/s against 22.6 serial. Fix: (1) a rollback that re-forwards the accepted prefix is a
second forward: capture only the compressor rings per token during verify (`DsparkAnchors`, 0.24 ms vs ~65 ms); (2) a
host read inside a layer loop is a GPU barrier: chunks that close no compression window defer their reads into one
eval (`chunkCrossesBoundary`); per-sub-chunk cache clears were wrong at spec widths. 1.56x serial. Measured against:
the confidence gate (our verify is fixed-cost dominated, ~70 ms + 12 ms per row, so a narrower block commits less for
nearly the same cost) ships off. A phase that is zero on some rounds must not be read as a per-round average.

### The prefill guard billed a dense attention shape for a sparse arch
Defect: a 5.8K prompt to a 1M-context model was refused at 10 GB. Cause: the score term billed `heads x chunk x seq`
for an attention that reads at most ~641 keys; the FFN width was Gemma's struct default (the config states none). Fix:
`prefillAttnKeys`, `prefillFfnWidth` (top-k x moe width + shared: under-billing is an uncatchable OOM, so the
asymmetry decides); later an arch-owned bill (`dsv4PrefillMemoryNeeded`) for the 512-token sub-chunk MLP and the f32
gathered-K set, reading the live sub-chunk width.

### DSV4 serial perf: the sorted gather 2x and the wrong barrier theory
Defect: a measured barrier cost at verify width (C≈6) was extrapolated to prefill chunks. Removing every barrier moved
nothing at C=512: the cost was `gather_qmm` with unsorted indices re-reading each layer's ~1.9 GB bank ~12x. Fix: the
`_gather_sort` pattern (prefill 2x); decode +17% from quantized `wo_a`, GPU emission, a lazy pipelined decode and hc
fusions; then `dsv4_dec_chain` (+8.6%: ~80 serial glue ops per layer into 4 dispatches). Lessons: a `metal_kernel`
helper owes its own `streamIsGpu` guard; a per-token-varying template value is a fresh JIT per value (ramp values ride
inputs); QAT sims are exact with `frexp`/`ldexp`; a repetition detector must stride 1. The fused gate+up and sink
softmax lost and stay opt-in.

### A shared lossy flag's default is per arch
Defect: comp_in int8 requant gave DSV4 +7-8% but duplicated a paragraph on one of seven characterization prompts. Fix:
it ships explicit opt-in (`decodeAttnQuantExplicit`) while Laguna keeps the flag on by default: adoption is not
inheritance.

### Stochastic DSpark acceptance
Defect: agent CLIs omit temperature (generation_config 0.6), so every agent request ran serial and only `--temp 0` got
DSpark. Fix: the one-hot-proposal arm of the MTP accept graph (accept with `min(1, p)`, residual correction from the
original verify logits) on a split begin/finish seam that the greedy path also rides; `dsparkArmFor` picks the arm.
2.0-2.3x on code and echo, parity on prose.

## Media DiTs and loop tiers

### Sparse video attention is dead by measurement (MiniMax-H3)
Defect: (null) an SVG/STA-style pattern measured 15x in a µbench, 1.33x live, and rendered smeared tiles. Cause: the
temporal arm puts adjacent spatial positions in different batch entries, a hole by construction; PSNR 9.8 dB and a
clip compressing at 10x the bitrate flagged it first. SDPA and linears both run within ~12% of the dense-bf16
roofline, so wall clock moves only by doing fewer forwards. Also found: the fast recipe's gate was a no-op at ≤ 6
steps: a schedule-scaled gate needs a test at the short end.

### The near-repeat loop tier convicted a voxel scene
Defect: a pagoda voxel scene was cut at 16,241 tokens and wrote no file. Cause: both ratios measure vocabulary, and
procedural code has a loop's vocabulary (0.068 and 0.351 against bars 0.12 and 0.35). Fix: a third ratio, novelty
(second-half 4-grams absent from the first half: scene 0.632, loops ~0.02), bar 0.10; all three must be low. Later a
span bar (4096) as well.

### A prefix-cache hit is bit-exact on attention and is not on a hybrid
Defect: llmprobe's determinism check saw LFM2.5 diverge between cold and warm runs at temperature 0. Cause: a
pure-attention restore replays stored KV (bit-exact); a hybrid re-runs the recurrence from the checkpoint in a
different block size; logprobs showed an exact tie at the divergence and agreement within 0.047 before it (bf16
scale). Not corruption. Rule: byte-stable long greedy on a hybrid needs `--prefix-cache-entries 0`; a byte-equality
test over one prompt cannot stand in for a determinism claim (short prompt + long generation exposes it).

### A borrowed-handle cache freed a live entry and killed the server
Defect: the `qwen3_5_moe` bench row scored 33% conformance and 0% capability: whole surfaces at 0/N means a dead
process (its log was a third as long and had no shutdown line). Cause: `cachedScalarInt` evicted slot 0
unconditionally, so past 16 distinct (K, N) values the N call freed the K scalar the caller still held; a comment
asserted it safe ("graph nodes hold their own refs": true, irrelevant). Fix: LRU eviction (`vqmmScalarEvictIndex`). A
bench harness reports whatever the engine returns: check cell logs for uniformity.

### The prefill-chunk cap reads the score width
Defect: an MLA arch scoring at nope+rope while `head_dim` stays the value width took the `<= 128` "fused covers it"
exemption. Fix: `ModelConfig.prefillScoreHeadDim`; the hd-256 measured branches stay keyed on 256 exactly.

### A per-arch spec exclusion written as a hand-rolled conjunct is a list of one
Defect: a second module-owned arch got none of the DSV4 exclusions: two concurrent requests shared one state and
`--pld` verified through it. Fix: one predicate, `ownsModuleDecodeState()` over named fields, and the pure
`specInitWiring`. An arch earns spec modes by shipping per-position capture plus a rewind, pinned at every accepted
count by error ratio (not bit equality: MLX picks its GEMM by M). A module-owned forward must honour
`ctx.capture_hidden` via `skip_lm_head`, or MTP drafts from a null handle.

### Kokoro: the SineGen no-op, load traps, the vocab invariant
Defect: waveform cosine 0.477 instead of 0.997. Cause: the reference's random initial phase at sample 0 is never read
after its 1/300 downsample; this port works at frame rate, where adding it is real. Caught by measuring the
reference's own seed-to-seed similarity (0.994): a stochastic component excuses thousandths, not halves. Traps:
safetensors read on the CPU stream; a depthwise `ConvTranspose1d` transposes `{0,2,1}`; `AdaIN1d` ships no affine
weights (pure instance norm); the vocab includes uppercase diphthongs and `Vocab.encode` drops unknowns, so the
invariant is "every emitted symbol is in the vocab".

## DFlash

### The assistant's precision is a per-round read; the A/B must not name its arms
Defect: DFlash read the whole 5.11 GB bf16 assistant and the trunk head every round. Fix: `DflashLinear` serves dense
or affine-packed from the checkpoint's own layout (a shipped quant is served as-is; an odd contraction dim stays dense
per weight); a draft-only head reuses `mtp.requantizeRows` (decide the dense arm before resolving a head's quant
params, or mlx-c gets a null scales). Measurement trap: a prompt nonce that named each boot's arm re-rolled every
always-thinking generation, so the arms did different work: a nonce varies by rep and is identical across arms.

### The block is a GPU decision, and it was worth 2x
Defect: DFlash at the config block of 16 was 1.01x on a 4-bit Muse trunk: the 16-wide verify cost 4.2 serial steps
(MLX's split-K qmm wastes half a tile; no lane covers 8..16 rows off G17). Fix: cap the block when no wide lane exists
(block 5: 1.97x); the assistant context rides the prefix cache (`DflashSnap`, adopt only on `base + step == matched`;
restores had dropped acceptance 92.6% → 66.5%). Closed: the draft-only 3-bit head (a win at 16, a loss at 5:
re-measure draft-side lossy defaults after the round cost moves), cross-round pre-draft, adaptive block, a wider
split-K tile. Sampled drafts lost (acceptance `1 - TV(p, q)` needs q close in shape); a batched `applyTopK` flattened
across rows (axis-less `mlx_topk`) and returned NaN: a helper that has only seen one row cannot reveal an axis bug.

### The sliding trim was decode-only, and its guard predicate had drifted
Defect: every verify block and prefill chunk read the entire cache on sliding layers; on Muse DFlash went from 2.28x
to 0.74x at 17.6k. Gemma got nothing at all: the in-kernel band predicate tested `seq_len >= 2` while the kernel
declines below 16. Fix: a block of `q_len` queries needs `window + q_len - 1` tail rows (`slidingViewFor`), the mask
built at the trimmed length (a mismatched mask silently attends the wrong slice), one shared constant, no width cap.
Laguna prefill 504 → 763 tok/s at 19k. A guard stricter than what it guards is a silent no-op. A prefill chunk trims
only at a non-zero offset, so test past two chunks and prove engagement by the `[sliding] block trim engaged` line.
Guard: `tests/test_tools_long_context.sh` (asserts the engagement line).

## Ling 3.0 (KDA) and fallible constructors

### KDA: a lower bound that replaces the formula, a gate that is an indexing contract
Defect: (port traps) `kda_lower_bound: -5` reads like a clamp but selects a different function (`bound *
sigmoid(exp(A) * g)`), which disagrees most mid-range; KDA gates per key channel. Fix: `kdaGateChain` pinned by the
properties a clamp breaks; `gdnKernelSource(vectorized, capture_seq)` generates all variants from one recurrence; a
uniform per-channel gate must reproduce the scalar kernel exactly; the blocked prefill kernel declines vector gates.
Trap: a cache struct inside a generic function is memoized unless it captures the comptime params, so four variants
shared one kernel. An absent bound routes to the softplus chain (`kdaUsesBoundedGate`), not to a bound of 0 (a gate
that never forgets).

### A cache that assumes one width
Defect: `--kv-quant` on an MLA arch (K 192 / V 128) hit an uncatchable mlx shape error: `updateAffine` sized every
buffer from the key. Fix: every buffer in every scheme is sized from the operand it stores. Guard: `KVCache affine
quant carries an asymmetric K/V too`.

### A fallible re-init behind a `deinit`
Defect: a correct load-time refusal printed and the process died in `mlx_array_free`. Cause: four sites did
`cache.deinit(); cache = try init(…)`, so the error left a freed cache installed and the owner freed it again. Fix:
`KVCache.reinit` builds, then frees and swaps; failure leaves the live cache serving. A constructor becoming fallible
changes every caller that frees before calling it. Guard: `KVCache dense update` fault sweep and the reinit test
against the testing allocator.

## Verify lanes and fused KV reads

### The verify-qmm plain-SIMD tiles were a 4-bit specialization
Defect: every non-4-bit pack fell to stock MLX on M1-M4 at verify widths. Fix: the tiles take `BITS` as a template arg
(MLX's kernel name includes the template hash, so no cache-key change), hoisting the weight words, not just the
pointers; adoption is shape-gated (`mixedPlainShapeEnabled`, N ≥ 5120): 6-bit 1.10-1.22x, while forcing every shape
gave back half; 8-bit stays off (≤ 1.04x). Cell order biases a paired sweep (~2%): reverse it on the second pass. Byte
identity with the lane off is not a property of any verify lane; the bars are fp32 dequant parity per width and 4-bit
bit-identity across the refactor.

### Fused kv-quant reads under spec decode: the verify widths were the whole bill
Defect: with `--kv-quant 8` MTP decode collapsed past 8k context (28 tok/s vs 60 dense); old llmprobe cards it was
compared against ran kv-quant off. Cause: verify widths (T_q 2..7) in the fused arm fell to a composed packed-qmm
chain per layer per round. Fix: the fused read is kernel-or-dense per width; a packed verify kernel for T_q 2..8
recovered kv8 ≈ kv-off on M4 (off on G17, where it lost); later the accumulators spilled past 12 rows
(`qkvVerifyRowsPay`), and a `matmul2d` kernel over a threadgroup-dequantized page (`qkvAttnMppKernel`, window
`qkvMppWins`) closed the gap (27B kv8 MTP 32K 37.1 → 50.8 tok/s). Gemma's gqa16 x dk512 layers overflowed the decode
kernel's staging and fell to the composed chain (-17%): the composed path never serves now; the per-layer floor rose
to 2048. Engagement, not output equality, is the assertion. Guard: `tests/test_kv_quant_fused_equivalence.sh`, the
parity cases.

### MLX's sdpa width wall at hd 256
Defect: dense causal verify blocks at q_len 6..15 ran MLX's slow fallback: its vector kernel serves only `q_len * gqa
<= 32` at hd 256 and our fused kernel floors at 16. Fix: `splitCausalSdpa` splits queries into groups of min(8,
32/gqa) rows against end-aligned key windows (byte-identical to consecutive smaller rounds), yielding to NAX
force-fuse past 8 rows: +4-9%. A per-width engagement log (a one-shot log only ever witnesses the warmup). Nulls: a
warm-at-real-KV warmup (our metallib is AOT); `--mtp-max-depth` is a cap, so force verify width with PLD draft-len.

### Dense bf16 MTP head trunks are requantized at load; spec byte bars need a tie-aware acquittal
Defect: a dense bf16 head pays its full read every draft step. Fix: `mtp.loadTrunkLinear` requantizes head trunk
linears to 4/g64 (fc stays bf16): +10% at equal acceptance. The equivalence test's byte bar then failed with the
requant off too: verify (qmm) vs serial (qmv) flips exact ties (`' canvas'` vs `' blank'` at a 0.0000 gap). The test
now boots with `--prefix-cache-entries 0` and acquits a mismatch only when the serial top-2 gap at the divergence is ≤
0.15 nats. A warm hit then left MTP history empty (fixed later by `MtpSnap`).

### The MTP committed history rides the prefix cache (`MtpSnap`)
Defect: every warm hit started the MTP head's history empty: ~38 tok/s warm vs ~72 cold. Fix: the history snapshots
onto the entry (`Entry.mtp`, the `DflashSnap` machinery). Commit trims the speculative tail first (`mtpCommittedLen` =
min of cache step, pre-draft and stash origins); restore is strict (declines when `matched > base + step`). Bar:
identical output plus a restore line and warm throughput, not identical round partitioning. Nulls the same day: the
GDN capture-tail audit (capture < 1% of a round) and a crossrow M 8/9 lane (-4%: stock qmm already reads each weight
once at those widths).

## Neural Engine prefill offload (`--ane-prefill`)

### What the ANE build taught (M4)
Defect: (port traps, M4 Max 27B: +12-18% prefill, later +18-26% in channel mode) `ANECCompile() FAILED` with no reason
on every layer, while the same program compiled standalone: the compile runs in `aned`, which cannot read
`~/.mlx-serve`, so staging stays in `$TMPDIR`. A single K=17408 down conv runs at 4.3 TFLOPS; K-chunking into slabs
restores 11.8. The share optimum is a measured hump. `msv_ane_model_eval` returns 1 on success. Parity: real weights,
fp32 truth, cos AND rms, then 16x amplitude probes. fp16 I/O planes are not bit-lossless (the compiler rounds earlier
on a bare fp16 output), so the bar is per-program parity + perceived-content equivalence. int4 weights are a no-go on
macOS 26 (every sub-byte constexpr form fails). Rules live in `docs/reference.md` "ANE prefill rules".

### The compile budget is the internal free disk, and a killed server leaks its staging
Defect: coverage fell 112 → 29 programs per boot over one night, and the server advertised `ready: 19/64` and ran an
A/B at 30% coverage. Cause: the content-addressed cache was never pruned (49 GB in an evening); a killed server never
freed its `$TMPDIR` staging (8-20 GB each); the compiler service keeps ~260 MB per program for the connection's
lifetime, so in-session retries cannot succeed. Fix: LRU prune (`MLX_SERVE_ANE_CACHE_CAP_GB`), pid-marked staging
reaped by the next process, compile inputs deleted after load (compiled artifacts must stay: aned demand-reads them),
a 1 GiB free-disk floor refused by name, and coverage that converges across boots via the cache. When ANE compiles
fail bare, check disk first.

### The share 0.35 "tiling cliff" was every eval failing
Defect: share 0.35 measured below the off arm. Cause: an fp16 plane's per-channel pitch (`rows x 2` bytes) must sit on
a 64-byte grid; rows ≡ 16 mod 32 compile silently and fail every eval, and the per-chunk fallback recomputed on the
GPU. Fix: `aneShareRows` floors to 32-row multiples and both emitters refuse other rows by name. Also: evals are
serial, so I/O planes are shared per shape class (bill 32 → 20 GB); the channel split (ANE holds output channels, GPU
rests are zero-copy slice views) matches row speed at 40% of the bytes.

### A built engine that never dispatched (MoE GDN-only coverage)
Defect: the 35B-A3B built 30 GDN programs and engaged zero. Cause: the tile was compiled at the pinned chunk (8192)
while the MoE forward chunks at 4096. Fix: the build resolves its tile through `effectivePrefillChunk`; +3.9% at 16k.
Built-but-never-dispatched is visible only to engagement counts.

### ANE prefill is M4-and-below
Defect: two M5 Max testers measured -11%/-7.5% at the shipped default: NAX GPU prefill outruns the ANE seam. Fix:
`ane.anePrefillAllowed` refuses on NAX by name (`MLX_SERVE_ANE_FORCE=1` overrides).

## DFlash2 and MTP heads in the wild

### DFlash2 port (incoai): names, a bare `qwen3` type, a reverted seam, a prompt gate
Defect: (port traps) the checkpoint uses z-lab's names (root `fc`, codebooks with no `.weight`); `model_type: qwen3`
registered the drafter as a chat model (discovery now asks the loader's contract predicate first); the trunk capture
seam had been reverted with an earlier experiment; the n-gram spec gate scored every novel prompt 0.000, so the
drafter never engaged and a test passed on template luck. Fix: probe both spellings, classify by contract, re-land the
seam, exempt DFlash from the prompt gate (the runtime yield gate cuts losses; bench bodies cannot carry
`enable_drafter`). Conv traps: `base_kernel`'s two leading dims are both 2 (a transposed reshape is silent); finish
kernels come from the sublayer input; position 0's predecessor is a zero pad. Muse's DFlash2 adds
`final_logit_softcapping` and `output_multiplier` that the selector needs. Drafter acceptance is a thinking-mode
property (12.5% off vs 47.8% on). Guard: `tests/test_dflash.sh` [3b], `tests/dump_dflash2_fixtures.py`.

### The Alis MTP head: a quantized `fc`, and a norm repair that broke a correct pack
Defect: Alis packs refused their head (`MtpTargetMismatch`), then decoded at 33% acceptance. Cause: `fc` ships
quantized and was compared by packed shape; the norm repair added +1 to an already-folded norm because this model's
backbone norms average 1.93. Fix: `fc` is a `QLinear` like every other head weight; the repair also requires the norm
itself to carry a negative value (a folded gamma is strictly positive). 70.5% acceptance after. Guard: `mtp: a
QUANTIZED fc loads verbatim and binds`, the repair-rule test.

### The DFlash block cap is a per-silicon table
Defect: `NO_WIDE_LANE_BLOCK_CAP = 5` was an M4 measurement applied to every non-NAX machine, blocking oMLX's measured
M3 Ultra block-8 win. Fix: `blockCapForMachine(chip)` keyed on the sysctl brand string (Ultra and Max share a GPU arch
name): M3 Ultra 8, else 5; `resolveBlockSize` takes the cap as a parameter (hermetic tests); the ready line names the
row.

### Spec snapshots ride the SSD tier too
Defect: a disk restore (fresh boot, post-eviction) handed back a warm trunk and a blind drafter. Fix: a
`spec.safetensors` sidecar per entry carrying the DFlash context and MTP history, sliced to each snapshot's step,
restored through the one `restoreSpecSnap` clamp; a size-mismatched spec is dropped alone (blind is valid, wrong is
not); geometry mismatches decline before `KVCache.restore` asserts. Guard: the round-trip tests,
`tests/test_dflash.sh` [12] across a real restart.

### M3 Ultra ANE: per-silicon share, an idle second ANE, machine-state spec collapse
Defect: the M4 default share was worth ~1% on an M3 Ultra; one ANE did all the work; a spec decode collapse survived
server restarts. Fix: `ane.defaultShare(mode, chip)` per silicon (M3 Ultra channel 0.35: +13.7% at 32k), `/props` ane
carries `evals` and `eval_failures` (zero evals on a green boot is built-but-never-dispatched). `powermetrics` shows
no ANE samples; `macpow --dump | grep ANE0_` does. A spec cell that collapses and does not recover on a fresh server
boot is machine state: reboot and re-baseline before attributing it to the build.

### Dual ANE: procedure banks and instance pinning
Defect: (port) naming an ANE die needs both `kANEFProcedureVariantHint` and `kANEFAneInstanceHint` at compile, load
and eval, or the hint is silently ignored; ~121 resident handles forced layers into banks of procedures; procedure N's
symbol indices come only from `procedureInfoForProcedureIndex:` (the other selectors return 0 for every procedure),
and identity indices made 200 of 210 evals fail into the GPU fallback, reading as "banks cost 23%". Fix: banks with
per-procedure indices (a multi-procedure bank is refused when the API is unreachable), a bank size cap
(the ~4 GiB device window and the build's host peak) with a halving ladder,
per-instance output planes, the total share split across units. Every arm of `tests/test_ane_prefill.sh` asserts zero
eval failures; an ignored hint looks like success, so `/props` carries per-unit evals and the IOReport cross-check
stands.

### A per-silicon MTP auto-depth cap, and a cap that fires silently is half a fix
Defect: on an M1 Pro auto MTP (cap 6) decoded at serial speed (10.64 vs 10.57 tok/s) while forced depth 4 ran 13.40:
verify width past 5 falls off a cliff there, and the controller maximizes accepted tokens per round. Fix:
`mtp.adaptiveDepthCapForMachine` (M1 Pro 4), which logs its row once (`[mtp] adaptive depth cap 4 (m1-pro row, default
6)`): without the line the symptom matches the controller or a flag. Superseded by the measured round-cost table
below.

### The verify-lane parity bar stopped referencing stock's worst element
Defect: lane parity went red on an M1 Pro (worst element 0.03 vs stock 0.004) and the first fix widened the slack per
chip. Cause: cosine cannot see one corrupted element; stock's worst element varies 0.004-0.036 across machines and
data. Fix: `VerifyQmmParity`: an absolute ceiling on the worst element (0.5), the lane's RMS error over stock's ≤ 3.0,
and the cosine pair; `verifyQmmParityVerdict` is pure and unit-tested.

### The armed spec flags vetoed batched decode, and PLD then turned itself off
Defect: four concurrent repetitive prompts decoded serially at 9.5 tok/s each with neither speculation nor batching.
Cause: `batchable` refused any slot whose request armed PLD; PLD drafted nothing and its yield gate turned it off.
Fix: `slotTicksRegular` reads dispatch (`specTickMode` + `spec_disabled_runtime`); 12.3 tok/s each.

### A batched group is bounded by padding waste, not slot count
Defect: one slot at 100k beside three at 1k built a ~6.5 GB padded per-tick transient no gate billed: an uncatchable
OOM. Fix: `groupKeepCount` keeps the largest prefix (by kv_len) whose padded tensor stays within `MAX_PAD_WASTE` 1.5
of the need (must stay below 2.0, the worst case of a pair); the longest slots decode serially the same tick.

### A batched-decode guard that only runs at N=1 pins a shape that never ships
Defect: `MLX_SERVE_FORCE_BATCHED=1` at one slot still has `batch == 1`, which arms decode-attn-quant and the fused
QK-norm gates that real N>1 does not. Fix: `tests/test_batched_equivalence.sh` runs a real two-stream arm and both
kernels log `[batched] ... engaged (slots=N)`; an arm whose curls did not overlap reports NOT-RUN.

### DSpark on LFM2.5: three silent zeros
Defect: LFM2.5 DSpark drafters loaded nothing, or engaged at near-zero acceptance, without an error. Cause: the
contract splits across root and `dflash_config`; a hybrid veto aimed at the Gemma drafter blocked every hybrid trunk
on four surfaces; DSpark's anchor row emits draft 0 (its `block_size` counts drafts); `rope_is_neox_style: false` is
MLX `traditional=true`. Fix: a `Contract` view (nested first, then root), one predicate `archBlocksAssistantSidecar`,
`anchor_row_drafts`, the Markov head (`markov_w1` gathered dense, `markov_w2` a linear): 30% vs 12% acceptance. A
sparse target takes an absolute floor of 1.8 accepted per round (a verify reads every routed expert). Rollback on a
conv-only trunk takes T explicitly. `lfm2_moe` dense layers come in two spellings (probed). Equivalence runs on
`/v1/completions`, where nothing amplifies a flipped tie. Guard: `tests/test_dspark_lfm2.sh`.

## The measured spec cost model

### A boot-time verify ladder could not decide widths (retired)
Defect: hand-typed per-silicon caps and four refits of the EV cost surface, each fitted at one context length. A
load-time probe of the verify ladder reproduced the M4 caps but decided nothing better, and its two extra terms lost:
a learned per-token KV term made deep positions look cheaper (the attention each query row does grows with context,
O(k x L)) and bought extensions nobody accepted (-2.7% at 21k); the fitted marginals under-priced depth 2.3x. Cause:
every width decision is accepted tokens over round time and a forward ladder measures cost alone (a forward is not a
round). Kept from it: a request-scoped learner can never see cross-request variation (state belongs on the model);
contention only adds time; skipped test arms read as passes when their model env is unset. The probe was retired for
the measured table below.

### The measured round-cost table
Defect: (design) no chip row is right across packs (peer sweeps put the best DFlash block at 8/6/none/5/4 across five
cells). Fix: `engines/mlx/spec/round_cost.zig`, a per-model table of round ms and emitted tokens per width and KV
bucket from solo single-chunk rounds, EMAs with stale-cell blending; the EV plan reads time from it; a width trial
explores. Traps found only by the simulated-loop test: extrapolating a cliff's slope downward priced narrower widths
at zero; a shallow slope extrapolated upward priced wide widths as free (past the widest measured, each position costs
max(last slope, prior marginal)); a struct literal assigning `m_lo` then reading it for `m_hi` aliased (build plans
from a scalar); one sample per cell let a transition-inflated round anchor the plan (`MIN_SAMPLES` 3, transitions
dropped, 5% hysteresis); a width ≥ 20% worse per token on its first sample is settled at once. The per-silicon row
stays the cold-start cap (`mtp_depth_free`). Persistence is opt-in, keyed on chip/model/quant/OS/engine build. The
DFlash `WidthChooser` stays opt-in: a MoE verify's cost is content-dependent.

### The MTP sidecar loader was affine-only
Defect: every nvfp4 pack (all of Ollama's MLX tags) silently ran without its MTP head: `missing tensor:
mtp.fc.biases`, one log line in a normal boot. Cause: `loadLinear` required biases whenever scales existed; the
forward passed the literal `"affine"` and solved geometry for groups 32/64/128 only. Fix: `quantParamsFromGeometry`
solves bits, group and mode from packed geometry (biases present or non-`uint8` scales mean affine); verify lanes stay
affine-gated. 29.2 → 61.7 tok/s on a 27B nvfp4 pack. Guard covers nvfp4, mxfp4 and mxfp8, not only the mode that
broke.

## Qwen3.8-Flash-Next (`qwen4_exp`) decode

### One f32 scalar widened the residual stream; a latency-bound kernel; an SSD fault per token
Defect: serial decode at 41.9 tok/s. Cause: (1) `mlx_array_new_float(0.25)` is a real f32 array, so the
hyper-connection inject weight and the whole stream ran f32 (the oracle passed at 0.99996 either way; `[dtype-trace]`
had said so in every log); (2) the fused hyper-connection read was slower than the 11-op chain: one lane walking 40
dependent loads is latency-bound, and a microbench with hot weights hid it; (3) the 32 GB n-gram table gather took 48
serial SSD faults per token (5 ms; `madvise` does nothing on Darwin). Fix: `scalarOf(v, dtype)`; `hcReadFused` (one
threadgroup per stream, unrolled loads, split-K down matvec, deferred write); a parked pool doing one `pread` each
(0.45 ms; faults on one mapping serialize on the VM map lock). 41.9 → 57.4 tok/s. Found by in-situ bisection with sane
stand-ins (zeros collapse the graph). The f32 fixture stays the math oracle; `qwen4 fixture bf16` pins the shipped
dtype.

### GDN decode fusion, three MoE nulls, and a profiler that was never off
Defect: (levers) the GDN prework and norm-gate fusion paid -3.9% (bit-identical per S except one bf16 ulp in 1 of 180
checks, which flips greedy bytes: legit). Folding the GDN in-projections, the shared expert as an 11th gather slot
(-3%: it ran in parallel off the critical path), and multi-row gather kernels were nulls or losses. The "70 ms MTP
verify" was the profiler: a harness exported `QWEN4_PROFILE_FWD=0` and the code tested presence. Fix: diagnostic envs
go through `diagEnvOn` (absent or `0` is off).

### The MTP acceptance audit: the gap was the prompt
Defect: first-draft acceptance 0.56-0.72 against the tech report's ≳0.8. Fix: a cross-round hermetic test acquitted
the bookkeeping (cos 1.00000); per-index acceptance on real packs met the report on code (0.91/0.80/0.68) and not on
prose: a property of the head, not the engine. One real finding: the prefix cache committed the qwen4 head's KV
without its state, so it is never committed or restored.

### MLX has no fused kernel for an hd-256 array mask
Defect: every QSA-masked attention past 2048 tokens ran MLX's materialized-scores path (steel skips hd 192/256 at q >
8; the vector kernel refuses `q * gqa > 32`; NAX requires causal with no array mask). Fix: a mask arm of
`msv_attn_p256` with a block skip table (+1-2.7%: 64 queries' top-512 blocks union to most of the KV, so it rarely
skips) and `splitMaskedSdpa256` for verify widths. The real lever was gathering selected blocks per query (below).

### "Module-owned" was a property of the pointer, not the state
Defect: qwen4_exp shipped serial because its module field sat beside DSV4's, yet it only holds read-only tables. Fix:
two lists (`module_owned_state_fields`, `module_shared_readonly_fields`); exclusivity moved to the slot driving the
MTP head (`slotExclusiveDecode`); PLE and QSA batched per slot. Two pre-existing GDN batching bugs on qwen3_5 too:
`KVCache.step` advances only on layer 0, which is linear on a GDN trunk, so batched tokens were roped at position 0
(offsets now from `moe_seq_offset`), and nothing advanced `moe_seq_offset` for a batched slot. A `mlx_squeeze` in the
gather arm dropped every singleton axis at B=2. 4 streams 1.73x. Guard: `qwen4 batched decode`,
`tests/test_batched_equivalence.sh`.

### qwen4 leftovers: every decode kernel was keyed on one row
Defect: a 2-row verify cost 1.5x a decode though not dispatch-bound: the fused hyper-connection and GDN kernels
declined at `batch*seq != 1`. Fix: row axes on the grids (`HC_FUSED_MAX_ROWS`, `GDN_FUSED_MAX_ROWS` 16), bit-identical
to stacked single rows; MTP on image turns rides the slot's M-RoPE table. `--no-mtp` did nothing on qwen4 (the
in-checkpoint head bound regardless): `entry.mtp` reads `params.mtp_enabled`.

### The qwen4 head planned M5 rounds with M4 costs
Defect: the Flash Next controller priced verify positions at the M4 default on an M5 Max, three to five times too
cheap: the first prose request ran 20-24% below serial. Cause: the `.qwen4` arm hardcoded `.generic`; a verify row on
this arch reads its own experts, so the marginal never flattens. Fix: a measured G17 surface behind a full
fingerprint, cap kept at 6. Trap: a cost-profile A/B with the
persisted round-cost table live measures the table: run persistence off on both arms.

### The grouped-expert NAX tile loses at verify widths
Defect: (null) MoE is ~60% of a verify row's marginal on M5, but a grouped-expert m16 NAX kernel lost: at verify
widths an expert run is 1-2 rows, and sorted `gather_qmm` is already near the expert-byte floor. The bytes are
irreducible without cross-row expert overlap.

### A 6-bit n-gram table that read as noise (#305)
Defect: a 6-bit `ngram_table.bin` dequantized to garbage. Cause: `dequantRow` assumed `32 / bits` elements per aligned
word; `mx.quantize` packs densely, straddling words at 3/5/6 bits. Fix: a u64 window across the straddle. A
hand-rolled unpack is tested at every width `mx.quantize` ships.

### QSA prefill gathers selected blocks; the mask arm walked every key
Defect: Flash Next prefill fell 1129 → 441 tok/s from 16k to 256k while oMLX stayed flat. Cause: a dense `[S, kv]`
bool mask (1 GB at 4096 x 256k) and a kernel that skips a block only when no row of a 64-row tile sees it (64 queries'
top-512 picks cover nearly every tile). Fix: `gatherQsa256` reads each token's sorted block list; selection in bounded
row chunks. 256k: 267 → 551 tok/s. Traps: loading the weights evicts the 32 GB n-gram table, so the first long prefill
faulted it in serially (`NgramTable.startWarm`); a stand-in that removes the indexer runs dense attention, not "no
indexer".

### QSA decode and verify read every row
Defect: decode fell 47.5 → 10.7 tok/s from 16k to 384k (dense mask + full-KV SDPA, and quantized KV dequantized whole
every step); verify widths did the same per layer per round. Fix: `qsaDecodeGatherAttn` gathers (and dequantizes) only
the selected rows: flat 51 → 41 tok/s at 32k-256k; `qsaVerifyGatherAttn` gathers a fixed-size superset of a verify
block's rows. Verify traps: the union tail is row 0's (tails are suffixes), duplicates double-count under softmax
(strictly ascending, distinct rows), and the parts must be disjoint; the floor was reasoned, then measured per KV
scheme (`qsaVerifyGatherMinKvFor`); `gatherQsa256` has no q_len floor, so a declined verify must not fall into it.
`mlx_take_axis` needs unsigned indices; quantized kernels misread strided take views (`takeContig`).

### QSA history in every SSM checkpoint OOM'd 400k prefills
Defect: Metal OOM at 400k while auto-context advertised 700k-1M. Cause: every stride checkpoint copied the growing
indexer key history (~30 GB). Fix: stride captures keep GDN/PLE only; one copy rides the latest snapshot and restores
slice it to the checkpoint's position (`applyQsaHistoryAt`), sliced onto the last kept checkpoint when a trim drops
the latest; the restore guard is `!=` (a history longer than the cache selected wrong blocks silently).

### QSA scalar RoPE skipped YaRN mscale
Defect: under a YaRN override, prefill and decode ranked different blocks. Cause: the indexer's scalar arm skipped
mscale on the reasoning that a positive constant cannot change a top-k: true for a full-head scale, false when only 64
of 128 dims rotate. Fix: one mscale on every arm (`yarnScaleRotated`); `--config-overrides` joins the SSD fingerprint.

### The n-gram gather synced on the token inside the graph build
Defect: 63 tok/s where the forward allowed 74: the PLE gather read the token ids on the host mid-build, so the build
never overlapped the GPU; the MTP verify build waited for the whole draft chain the same way. Fix: `ple_defer` hands
the graph a leaf and `flushDeferredPle` fills it after the build. Trap: a deferred gather on a verify must claim its
rollback slot at build time (`pleClaimSpecCapture`), or every partial accept restores the wrong PLE state silently.
The remaining step is ~6.4 ms of weight bytes and ~840 dependent kernels at ~10 µs: only kernel-count cuts that do not
redistribute work can pay (a fold that recomputed RMS stats in every threadgroup cost +1.3 ms). A stand-in whose
output nothing consumes is dead code to MLX.

### The qwen4 head: projections it never used, a cold EV controller, an EOS entry token
Defect: the head projected all `1 + accepted` rows to use one and drafts read the whole 675 MB 8-bit lm_head for one
argmax; every request restarted the EV controller cold; a round whose entry token was EOS drafted, verified, and lost
the prefix commit. Fix: `Qwen4MtpProject` and the shared coarse-shortlist rerank (the lm_head's input is the mixer
output, not the pre-mixer stream: `StepWant{logits, mixed, none}`; the coarse head builds lazily); the EV seed on the
loaded head; every block decoder checks its entry token first (`generate.tokenStops`). Two `errdefer` frees of one
handle at one scope double-free on the first error. Auto mode is not byte-reproducible (depth follows measured times),
so the byte bar is a pinned depth; `[spec-stats] depth=` is the cap, not the chosen depth. A binary copied out of
`zig-out/bin` cannot resolve its `@rpath`.

### Planning the base depth from acceptance lost; the horizon was the lever (null)
Defect: llmprobe `spec:predictable` alternated 111 and 154 tok/s on one binary. Cause: the base depth sat at 1 every
boot; predictable throughput came entirely from the extension horizon, which flapped (4.42 vs 1.86 accepted per
round). An acceptance-planned base (parked) lifted the base, which closed the horizon sooner, and cost 4-10% on novel.
An A/B on a bistable path needs pairings with the persisted table restored before every boot (the live table is the
subject).

### Speculation never compared itself with the serial token it replaces (`MtpAdaptive`, #363)
Defect: on qwen4_exp at 62.7k-374k, MTP ran below serial (30 vs 47 tok/s) with acceptance far above the floor: a
verify row is expert bytes. Fix: per KV bucket, both the table's ms per token and the request's trailing price window
must lose to a measured serial cell (a bounded 8-token probe teaches an unmeasured bucket) past a 32768 floor; the
switch is sticky on a module-owned head, which is released so other slots batch. A modeled token denominator
under-predicted by 12-31% and switched where MTP was faster. `--max-mtp-ctx N` is the operator ceiling. Guard:
`tests/test_mtp_adaptive.sh`.

### The round-cost grid and store version belong to the arch (#363)
Defect: the long-context bucket grid shipped for every arch, so every sidecar pack booted a cold table, every round
planned two chunks (-25% at 8k on the 27B), and a cold bucket inherited a neighbour's trial period for 45-91 rounds.
Fix: `round_cost.Layout` per model with legacy migration; `TrialSchedule.force` re-reads the period every round.

### The qwen4 MTP head's state is not KV-only, and its row count is its step
Defect: a persisted head restored without its QSA key history errored `QsaHistoryGap` and drafted blind on every hit;
then it persisted with step 0 (`KVCache.update` advances step at layer 0, and the head's layer is
`num_hidden_layers`). Fix: both halves travel together (`qwen4MtpAdopt`); `qwen4MtpAdvance` sets the step and the
commit refuses a mismatch.

### QSA trim budgets include the transferred pooled bank
Defect: a 496k commit trimmed to a prefix billed 56 MB, then cost 11.9 GB against an 11.7 GB cap and evicted the only
entry. Cause: trimming moves the pooled indexer bank onto the kept checkpoint. Fix: `trimmedCheckpointBytes` prices
the sliced bank at each candidate.

### QSA at long context: split-K, an exact select, one history copy (#363)
Defect: verify widths spent ~50 dependent ops per layer; the selection's `b * 1e-7` tie bias vanishes above |score|
~0.84 (production magnitude) so two slots with equal scores attended different blocks; an all-visible skip handed a
null sheet to `mlx_logical_and` and aborted on the first token past kv 2052; the indexer history lived twice through
decode (3 GB at 786k). Fix: one split-K dispatch with in-kernel unpack (`QsaCfgCache` keyed without runtime shapes);
`msv_qsa_select`, an exact radix select on every arm; the skip gated by `nb > block_topk` (test kv 2049..2052); the
newest checkpoint takes a slice view of the slot's buffer at commit, and the f32 score bank is billed; the score-sheet
budget is billed to admission. The select ran one threadgroup per row, which decode starves: split 16-way at decode.

### The pad-waste cap priced every hybrid slot at kv_len 0
Defect: on linear-layer-0 trunks `cache.step` is 0 forever, so the waste ratio was 1.0 and a 1k slot beside a 60k one
padded 59k rows per tick. Fix: `KVCache.kvLenForBatching` reads the first attention layer's offset.

### PLE prefill measures table reads, never infers residency
Defect: a cold 29.8 GiB n-gram table gave 1.6-3.0 s TTFT after startup though the KV was short; one slow sample later
put a warm table on the slower pool (#821). Fix: `NgramTable.calibrateArm` measures both read arms on the inference
thread before and after warming, median of 5 trials; explicit overrides win.

### A contaminated round-cost cell that no trial could re-measure
Defect: the 27B pack benched 51 tok/s on a day it did 66: a persisted w2 cell read 3x w1, so the plan never left width
1, and a measured cell is never a trial target. Cause: rounds interleaved with other slots' prefill chunks carried
~100 ms each into the round clock; then a cell poisoned in tokens (echo-era vs prose-era samples); then a poisoned
width-1 cell nothing narrower could check. Fix: interleaved prefill drops the round clock; folds and loads refuse
samples past `IMPLAUSIBLE_STEP` 1.5x of the nearest trusted narrower cell, `SELF_SPIKE` 3x of a mature cell, and (load
sweep only) `IMPLAUSIBLE_WIDER` 1.25x of the nearest wider cell; per-token cost divides by the monotone envelope
`planTok`. Uniform contamination is invisible to ratios: delete the table. The "predictable collapse" that started it
was the 27B refusing llmprobe's repeat prompt on half its cache-bust tags.

### Batched MTP verify: eight rows fall off the split-K lane; a crowd beats sub-groups
Defect: two MTP users decoded slower together than one alone. Fix: a group verifies in one `[N, S]` forward (rows
padded, per-row capture split back); the first cut was 2x slower because 8 rows left the split-K lane (groups capped,
later the wide shader tile); past the crowd threshold slots decode plain; the qwen4 head's state moved to a
per-request `Qwen4MtpState` swapped in before every touch, batched verify opt-in there. Guard:
`tests/test_mtp_batched.sh`.

### The QSA score sheet and the gather attention
Defect: (levers) the indexer score ran four memory-bound ops per row chunk over a 256 MB f32 sheet (~3 s per prefill
chunk at 1M); the gather attention ran at ~3 TFLOP/s. Fix: `msv_qsa_score` reproduces MLX's tf32 k-loop order and the
sequential 4-head sum bit for bit (every operand is already bf16-valued); `msv_qsa_nax_precise` (PR #385) runs the
gather on NAX cooperative tensors (macOS 26.3 + G17, its own predicate `qsaNaxEligible`), softmax weights split into
two bf16 terms (two match stock's error, one is worse): +12.6% prefill at 162k. Bar: error vs float64 no worse than
stock (`tests/qsa_nax_precision.py`). The score must stay on the NAX kernel at every width so decode, verify and
prefill pick the same blocks.

### A 3-row conv_state view pinned the whole prefill chunk input (#366)
Defect: every conv-cached layer kept its whole chunk input alive until its next forward (3.0 GB on qwen4_exp at width
4096, 7.5 GB on the 27B). Cause: `conv_state` was a slice view; a lazy owned copy is not in the residual's graph, so
an eval of the residual leaves it lazy and still pinning. Fix: the copy is named in the in-loop cadence eval
(`evalCadencePoint`), gated on chunk-wide parents (T ≥ 32); the admission bill (`prefillStreamBytesPerToken`) dropped
the pin it had been billing. `mlx_array_equal` cannot see `-0.0 → +0.0` from an `x + 0` copy; live greedy identity
covers it.

### A failed dense KV write left freed view handles in the entry
Defect: SIGSEGV in `freeKVEntry` after a caught MLX error during an embedding sub-batch write. Cause: `updateDense`
freed the old views first and assigned new ones only after the writes. Fix: reset handles at the free; `writeAtOffset`
releases its result on error. Guard: the `KVCache dense update` fault sweep.

### 3-bit experts fell off the fused MoE decode kernels; the fix is the pack unit
Defect: the 3.3 bpw Flash Next pack decoded slower than the 4-8 pack while reading 30% fewer bytes: every fused MoE
kernel declined 3-bit. Fix: the pack unit is the byte triple `mx.quantize` writes, read through one
`mlxserve_qpack<BITS>` loader (a 32-value unit left lanes idle). At ≤ 4 bits these kernels are ALU-bound, so a
narrower pack pays only with balanced lanes.

### The MoE down+reduce kernel was reduction-bound
Defect: the down kernel read half of gate+up's bytes in the same time. Cause: K is the MoE intermediate (640), so a
whole simdgroup per row paid a 32-lane reduce with lanes idle. Fix: eight lanes per row, packs hoisted per lane, one
shuffle reduce for four rows: 34 → 24 µs; parity moved to RMS vs f32 truth. Separate test runs drifted 15% between
identical kernels: A/B kernels interleaved in one process.

### Prefill fusion must preserve the compiled bf16 chain; a template width is a JIT per prompt length
Defect: an eager reference hid rounding changes MLX compilation introduces; the first HC/GDN prefill fusions carried
the chunk width as a template arg, so every novel prompt length paid three JIT compiles (a loss below ~1k tokens,
invisible to llmprobe which repeats lengths). Fix: guards compare against the compiled chain (bf16 mean reciprocal, a
shared MLX-generated sigmoid table); the width rides as a 0-dim int input.

### Sampling block helpers: one filter, one accept per group, a type for committed argmaxes
Defect: a group's sampled MTP accept ran N serial one-row filters with the GPU idle across host reads; top-k then
top-p selected twice; every greedy verify argmax could commit a reserved or padding id. Fix: `mtpGroupSampledAccept`
filters the joined verify block once (declines on mixed samplers, seeds or padded rows); `filterTopKTopP` is one pass
(byte-identical); `verifyArgmax` masks first and returns a `CommittedArgmax`, the only type a commit site takes. A
name-keyed source scan written first missed the grouped path's `var am`.

### A best-effort sidecar write failed the next request
Defect: ~2.6% of chat requests 500'd: `expected a non-empty mlx_array (map.cpp:49)` → `spec persist failed` → the next
tick's MLX failure. Cause: a pooled-only QSA head passed an OR guard and inserted an empty array; the throw was
caught, but the process-wide latch failed whoever ticked next. Fix: the guard requires the array; `writeSpecSidecar`
drops the latch it raised (`dropLatchedErrorUnless`). A swallowed MLX error is not swallowed until the latch is
cleared.

### A warm restore forwarded the 31-token tail as 30 + 1
Defect: stream and non-stream differed when the second request hit the cache. Cause: cold forwards the last 31 rows as
one span; warm restored at the backoff snapshot and ran 30 + 1, different tilings, a flipped tie. Fix:
`ssmSnapshotBackoff(…, restored)` forwards a restored tail inside the window as one span. Guard:
`tests/test_hybrid_reuse_equivalence.sh`.

### A group-padded verify row broke the sampled accept (#446)
Defect: any sampled row padded past `1 + m` raised inside mlx-c and failed the whole group with a 500. Fix:
`verifyRows2d` slices the first `1 + m` rows (a short block is a named error). Guard: `batched corrections read only
1+m rows of a group-padded verify block`.

## Exact decode, kernel geometry, concurrency

### 2-bit GEMV: half2 sums were fast and not exact; a one-chip geometry lost elsewhere
Defect: the Bonsai GEMV accumulated in half2 (1.55x stock's RMS error under f16, subnormal prescale); later the
M4-tuned geometry ran 0.85-0.9x stock on M1 Ultra and M5 Max. Fix: exact half2 decode, f32 sums (codes and x kept
half2 in registers, widened at the FMA); `msv_qmv2_rows` (M 1..8, bias dropped for ternary) with `planFor` per GPU
generation from sweeps on each; an unmeasured generation keeps the old dispatch. A tolerance vs stock's output passed
the half2 kernel; the bar is no worse than stock vs f32 truth.

### DFlash beside company decoded serially
Defect: with a DFlash sidecar, four streams aggregated below one (64 vs 74 tok/s). Fix: `requestSpecModes(…,
has_company)` hands the round to the MTP head when others are in flight (122); later the first request of a burst
yields after 2 ticks with company (`specYieldTick`).

### Small dense models spent a third of a token on dependent launches
Defect: 4B-26B models decoded at 55-65% of the bandwidth floor (~940 kernels per 4B forward, ~3 µs idle each). Fix,
all bit-identical: q|k|v and gate|up rows joined at load at decode width only (`fuseRowGroup`); one residual kernel
with MLX's own `rms_single_row` tree (`fusedResidualNorm`); unary activations tabulated over all 65536 patterns
(`tableGateMul`); the norm+RoPE kernel at hd 256 with v rows; LFM's gated conv step; Gemma's PLE tail with an
op-for-op `qmv` replica. +4-10% decode. Joining independent matmuls alone and cutting barriers did not pay; a base-3
ternary repack read 18.75% fewer bytes and ran 20-30% slower. Guard: the fused-op tests and byte-identical transcripts
on 17 packs.

### Drafted output differed from serial, and seeded output from itself on a cache hit
Defect: speculative rounds at T=1 with a seed differed from serial; a seeded request changed on a prefix hit. Cause:
width-dependent reduction order in verify rows, the GDN window's f32 state vs serial's per-token bf16, differently
keyed draws, a sampler position counted from the uncached suffix. Fix: row-exact kernels ported from TensorFold
(`rowqmv`, `simd_qmm`, `row_attn`), per-token state rounding, `keyed_sample` at the absolute position, `position_base`
= full prompt length; scoped to DFlash-bound row-exact archs (`rowExactDecode`). Trap: `--no-mtp` still runs PLD.
Guard: the width-invariance tests, seeded cold == warm.

### A draft tree failed the second request under `--kv-quant 8`
Defect: with a tree drafter and `--kv-quant 8` (the app default), the next request 500'd `SpecTreeUnsupported`:
`compactRows` refused quantized KV, and only rounds whose accepted path was not the first branch compacted. Fix:
affine scales and biases move with their rows; exact decode never reads packed KV (the packed kernels ignored the tree
mask). Guard: `KVCache.compactRows: …`, smoke `drafter_kv8`.

### A 1024-thread threadgroup failed on CI, and 512 fails on M1/M2
Defect: CI went red with one MLX error and four collateral failures (a later segfault on an `undefined` Slot). Cause:
Metal caps threads per threadgroup per compiled kernel by register use (M1/M2 below 1024 and 512 for this kernel); the
latch was invisible because logging is off under test. Fix: probe at creation and halve simdgroups in the same
summation order (`mma_sg`), else decline by name; the test drops its own latch.

### Agents looped once prompt lookup ran under typical acceptance
Defect: Flash Next agent runs under `--mtp-typical` were loop-cut 11 of 19 times vs 0 of 4. Cause: a lookup draft is a
point mass, and typical acceptance keeps anything above `0.2 * e^-H` without a draw, so self-echo became certain. Fix:
lookup rounds verify with exact acceptance whatever the mode.

### A 1- or 2-node draft tree committed an unwritten conv row
Defect: a tree of fewer than 3 nodes gave different bytes from drafter-off. Cause: the prework grid's TL threadgroups
copied the conv window's 3 state rows only from `t < 3`. Fix: each threadgroup copies rows `t, t + TL, …` below 3.
Guard: `gdnDecodeTreeCase` at 1, 2 and 8 nodes.

### The DFlash yield gate sent a winning drafter to plain decode
Defect: a block-16 tree drafter at 1.9 accepted per round was disabled while emitting tokens 2.7x cheaper than serial.
Fix: `checkDflashRuntimeGate` asks the measured round-cost table first (`roundBeatsSerial`); the constant decides only
until both cells have samples.

### A kernel config cached by row count handed a 16-slot tick a 16-wide verify's shape
Defect: 16+ concurrent streams on a drafter-bound 27B failed every stream (`[concatenate] … (16,3,10240),
(1,16,10240)`). Cause: `add_norm` keyed its config on `B*S`; `[1,16,D]` and `[16,1,D]` collided. Fix: `CfgKey` carries
`b` and `s`. Guard: `addNorm returns each call's own [B,S,D] layout at one row count`,
`tests/test_batched_past_block_width.sh`.

### An image in any stream dropped the whole batched group to the dense mask
Defect: Flash Next behind a screenshot-attaching agent lost most of its aggregate decode at three or more streams.
Cause: one M-RoPE slot made the batched setup refuse the QSA gather arm for the whole group. Fix: the gather reads no
rope tables (queries are rotated, cached keys carry positions), so M-RoPE slots take it. Raw BF16 n-gram tables
(`bits=16, group_size=0`) also failed a group-size check that belongs to the quantized branch.

### A per-round buffer sized by one drafter's cap overflowed when another fed the round
Defect: MiMo segfaulted on a garbage `vision_embeddings` handle. Cause: the MTP stash held `MAX_DEPTH + 1` ids, prompt
lookup committed up to 14, and ReleaseFast wrote past it into `ForwardCtx`. Fix: `MAX_ROUND_DRAFTS` = the max over
every round producer. Tell: a segfault on a field nothing writes at a small-integer address: rebuild ReleaseSafe and
replay. Guard: `tests/test_mimo_v2.sh` [9].

### A full-buffer sliding layer caches every token it never reads
Defect: 87% of MiMo's 222 KB/token KV was sliding rows attention never reads; at the app's 1M context the bill left no
hot cache and an 88k agent turn re-prefilled every turn (~38 s). Fix: `slidingRing` keeps window + slack rows per
sliding layer (`KVCache.updateSliding`, absolute `offset` via `entry.base`), billed once per slot; a ring cannot
rewind past dropped rows, so restores stay above `ringFloor` and rings never spill to SSD. Guard: `KVCache sliding
ring`, the prefix-cache and DiskTier ring tests.

## GLM-5.3 and Flash Next kernels

### A prefill-tiled kernel at decode width is one threadgroup's work
Defect: GLM decode fell 54 → 35 tok/s: the hyper-connection mixes kernel streamed 1.5 MB through one threadgroup per
decode row, and the NAX indexer padded one query to a 64-row tile. Fix: the mixes kernel serves 512+ rows
(`MIXES_KERNEL_MIN_ROWS`); one token runs `hcPre` (16 K-slice threadgroups); one query runs `decodeSelect`.

### Partials crossing threadgroups in one dispatch went stale
Defect: `hcPre`'s last-arriving threadgroup read wrong partials, differently each run, even with fences. Cause: a
core's L1 can hold stale lines of a reused buffer. Fix: atomic store/load (through L2) and a fence before the arrival
increment; the counter is 8 words (a shorter input binds in constant space). Guard: 32 re-dispatches, bit-identical.

### An ablation that leaves an output unwritten measures NaN routing
Defect: skipping GLM's Sinkhorn gates "saved" 0.5-1 ms. Cause: garbage activations route every MoE layer to the same
cached experts. Rule: ablate with sane values (finite, normalized, in the original dtype).

### A cache row nothing reads in this forward stays a lazy chain
Defect: below 2051 tokens nothing read GLM's indexer cache, so decode (which evaluates token and logits only) never
computed its rows; the chain grew every token and stalled at 2051. Fix: the attention output depends on the cache
update (`mlx_depends`). Tell: a decode graph with no `SliceUpdate` for a cache the layer writes.

### GLM-5.3's long-prompt output moved with the chunk width, and it was not a bug
Defect: (suspected) a cold 9.5k prompt's top tokens moved 1-4 nats between prefill widths, and the prefix cache's tail
split flipped a greedy answer. Cause: rounding order on low-margin tokens: an equivalent kernel in one pass moved the
token as far as chunking did. Rule: byte-stable output on a hybrid needs `--prefix-cache-entries 0`.

### Flash Next serial decode is kernel-time and dispatch bound
Defect: (measured) an 11 ms step against a 5.4 ms byte floor on the M5 Ultra; three dispatch-count fusions were nulls
(the gated shared expert on the gather kernels made the step slower: a one-expert GEMV there is latency-bound; folding
already-overlapped elementwise ops is free). A Metal System Trace showed the big GEMVs at bandwidth and ~150 small
GEMVs launch-bound: a custom split-K small GEMV lost to MLX's `qmv` however laid out, and bigger command buffers lost
the head/tail overlap. Rule: only a kernel that replaces several dispatches and is no slower itself pays. Kept: a
register-resident router top-k.

### The one GPU idle per serial token was the n-gram history settle; an async batch has one event
Defect: the GPU PLE arm idled each step while the host read the token to advance the n-gram history. Fix: the record
stays pending and four sites settle it (next forward, batch join, spec drain, finish). It first lost 8%: every array
of one `async_eval` carries that batch's end event, so waiting on the sampled token waited for the whole forward;
`lazyForward` now ships the token as its own batch, read without queued ops. +1.5-2.5%.

### The routed-expert kernels: the fp4 shape, and rows per simdgroup is an occupancy decision
Defect: the per-slot gather kernels were kernel-time bound (down+reduce latency-bound at K=640). Fix: `moe_affine4`
ports `moe_fp4`'s shape (x in registers, 8-byte loads, no threadgroup memory). Then the down kernel looked
latency-bound and three fixes tied or lost (double-buffered weights -6%: registers); what won was one row per
simdgroup instead of four: 110.5 → 114.8 tok/s. Sweep rows per simdgroup in situ before deepening a multi-row kernel;
a sweep script must word-split its configs explicitly (zsh does not split `$cfg`, and the kernels declined silently).

### Long-context decode: the attention layer is a chain
Defect: past the indexer budget each serial step cost ~1 ms more in a chain of dependent launches. Fix: one kernel
writes the row's key mask, then `qsa_decode.attend` (split pass + merge over the picks, no mask/gather/SDPA): +4% at
8k-40k. MLX's concurrent encoder puts a global barrier before any kernel reading an outstanding write, so branches
overlap one stage only (ordering the q projection behind the indexer made the layer worse). The score kernel stays on
the matrix unit (no scalar order reproduces it). A reduction inside divergent code returns garbage: reduce, then
select.

### An embedded PLE table does not identify the norm convention
Defect: a Qwen4 pack with embedded PLE shards and already-folded norms would add 1 again, silently. Cause: table
storage chose the norm transform. Fix: unmarked embedded packs infer the convention from the indexer q/k norms
(unanimous means near 0 or 1); ambiguous anchors fail by name; a pack-local `qwen4_norm_convention` marker selects
explicitly; a server-wide override is refused.

### A fast path keyed on one pack's quant layout is silent on another's
Defect: oMLX's Flash Next oQ4e pack answered correctly at half our pack's speed (21.7 vs 11 ms): nearly every tuned
path declined without an error. Cause: kernels read the inject and shared gate as dense rows (oQ4e quantizes them),
unpacked 2/4/8-bit g64 only, verify lanes were gated on one width, `--ple-gpu` read one merged file, config caches
ignored the width (135 rebuilds per forward). Fix: `dequantizeSmall` for tiny dense-only tensors, kernels unpack every
affine width, caches key per width, gates re-measured per shape, embedded n-gram shards repacked into one buffer: 10.4
ms. On the dual-die M5 MLX's qmv batch limit is 18 rows (`qmvBatchLimit` mirrors MLX's table). Guard:
`tests/qwen4_engagement.sh <pack>` before measuring any new layout.

### A short MoE prefill re-read every expert per row
Defect: a few-dozen-token Nemotron prompt took longer in the MoE than one twice as long. Cause: the sorted gather
streams each expert once only at ≥ 4 sorted rows per expert. Fix: pad rows spread over the experts up to 4 each
(`moeStreamPadRows`), quantized banks on NAX only.

### A restored KV entry had no read views
Defect: GLM hot-cache hits with the MTP head's snapshot died intermittently (`expected a non-empty mlx_array`) under
10-12 streams. Cause: `KVCache.restore` rebinds buffers but only `truncate` rebuilt views; the head's snapshot matches
exactly, so nothing truncated, and GLM's DSA reads the cache before its first append. Fix: `rebuildViews`, and
`denseView` rebuilds empty views. Guard: `KVCache restore: a restored entry exposes its live rows to a read before the
first update`.

### Joint-projection rows fed to the DSA core corrupted it (open)
Defect: per-slot views of GLM's batched DSA input projections gave cos 0.2-0.4 against serial; evaluating the joint
arrays first fixed it. Cause unproven (a lazy-graph hazard around sliced views feeding the pooled-rows kernel). The
shipped path projects per slot; do not retry without a bit-level test.

### GLM's verify window ran the KDA op chain silently
Defect: the KDA decode step never engaged at 2-8 rows. Cause: it needs the row-joined projection, handed out at width
1 only. Fix: a KDA window takes the joined projection. Tell: no `[kda] decode step engaged (T=N…)` line at a verify
width.

### A grouped verify drifted from its solo verify (#727)
Defect: `every row of a group verify is byte-identical to its solo verify` failed from row 1. Cause: the fused GDN
verify arm declined joined projections, so groups ran the composed chain; and joined groups past 8 rows leave the MoE
rows arm. Fix: the joined path takes the fused arm; the MoE difference is accepted (joining is faster), so the test
bounds its scope.

### DFlash2 over Bonsai: dtypes and a load peak
Defect: a bf16 drafter over the f16 Bonsai 27B ran slower than MTP and OOMed on 16 GB. Cause: f16 inputs met bf16
weights (every op promoted to f32, `simd_qmm` declined); the loader quantized lazily and held the whole bf16
checkpoint. Fix: `toDrafterDtype` at the drafter's entries, per-linear eval + `Weights.replace`: peak 12.5 → 9.4 GB,
22.5 tok/s. Tell: `[dflash-trace]` assist far above bytes / bandwidth.

### Every tool round re-forwarded the previous turn's generated tail
Defect: 98% of Flash Next agent rounds re-prefilled the 31-token backoff plus the whole previous reply. Cause: a
hybrid restore lands only on an SSM checkpoint, and all were prefill checkpoints. Fix: the commit appends a checkpoint
at the committed length when the live state is exactly that prefix (`decodeEndCheckpointWanted`), with the MTP head's
one-row-short history carried (`specCarriesOneRow`); a plain step forwarded a lookahead, so it rolls back first
(`rollbackSsmTo`). Guard: `tests/test_cache_reuses_generated_tokens.sh`.

### A compiled region re-read its captured weights from disk every call
Defect: DSV4.1 on a 152 GiB pack spent five minutes in warmup, then OOMed at ~200 GiB. Cause: `mlx_compile`d per-layer
closures captured lazy safetensors loads, so every call re-read weights into fresh buffers. Fix:
`deepseek_v41.evalWeights` evaluates every array the model reads at init. A closure that captures arrays needs them
evaluated before its first call.

### A pack near the RAM size was compressed while it loaded, then killed
Defect: a 231 GiB pack on a 256 GB Mac died `Killed: 9` during load ("out of application memory"). Cause: MLX's reader
left shard pages in the file cache, and buffers become wired only when a command buffer runs; the kernel compressed
148 GB. Fix: `nocache_reader` (F_NOCACHE, aligned stages) and a one-op command buffer after each layer (`wireLoaded`).
Tell: `vm_stat` wired flat while the footprint climbs.

### The DFlash policy's plain cost came from a table that folds outliers
Defect: with the round policy on, novel text decoded below serial and `skipped=0`. Cause: the plain cost came from a
round-cost cell that one slow tick moved 16.4 → 27.1 ms. Fix: `dflash_policy.PlainCost`, learned from the request's
own consecutive ticks, moving at most a quarter per sample.

### A bare `@min` against a constant is a tiny integer
Defect: a 15-draft copy chain SIGBUS'd the server silently. Cause: `@min(p.len, MAX_DRAFTS)` with a comptime 15 is
`u4`, so `n + 2` overflowed (no check in ReleaseFast). Fix: type the result (`const n: usize = @min(…)`). A silent
ReleaseFast exit: read `~/Library/Logs/DiagnosticReports/*.ips` first.

### A Vulkan backend's MSL translator refused our kernels at eval
Defect: on the mlx-omarchy Vulkan backend every request 500'd in prefill. Cause: custom Metal kernels ran on every
non-CUDA backend, and the translator refuses at eval, past every decline probe. Fix: `metalKernelsAvailable` is
Metal-only; MTP is off by default wherever the verify kernels cannot run.

### The standard forward handed back logits under `skip_lm_head`
Defect: the forced aligner's head received the trunk's logits (an MLX shape error). Fix: `forwardStandardWith` honours
`skip_lm_head` (the LTX text encoder had paid a discarded full-vocab head).

### Weights on the CPU stream lived in host memory on Windows
Defect: on Windows and WSL Qwen3-0.6B decoded at 28 tok/s against mlx-lm's 130; one qmm took 270 µs vs 10. Cause:
without concurrent managed access MLX backs a stream-less allocation with `cudaMallocHost`, which never migrates, so
every token re-read weights over PCIe. Fix: `mlx.weightLoadStream()` (GPU stream where memory does not migrate) for
every loader and the SSD restore; memory gates read VRAM there.

### Two cuBLAS copies left llama.cpp on the CPU
Defect: a GGUF on Windows decoded at 7.6 tok/s when an MLX model loaded first, 80 otherwise, no error. Cause: MLX
loaded its toolkit's `cublasLt64_13.dll` and `ggml-cuda.dll` failed beside it, falling back to CPU. Fix: no CUDA DLL
ships; both engines bind the installed toolkit's copy (`patches/mlx-cuda-windows-toolkit-dirs.patch`,
`exportCudaHome`).

### A rebuild, then every app update, recompiled every Neural Engine program
Defect: a cached ANE model recompiled for minutes after each build or app update. Cause: the cache variant hashed the
engine build id and the prune deleted other variants; the ANE service keys programs by the compiling process's
code-signing identifier, which codesign derived from the binary's hash. Fix: the variant names only configuration and
`rt.cacheSweep` drops untouched entries after a build; `mlx-serve` embeds an Info.plist whose `CFBundleIdentifier`
codesign uses. Check: `codesign -dv` on a re-signed copy prints that identifier (no automated guard).

### Bonsai 2 2B/4B/9B answered in gibberish: a decode kernel sized for one pack's block
Defect: Bonsai 2B, 4B and 9B produced word salad from the first token; the 27B was fine. Cause: each pack picks its
own Hadamard block (512/4096/1024) and the fused one-token GDN kernel was written for 1024. Found by matching prism's
loader layer by layer under diag `layer-cap`. Fix: the kernel takes the block as a template. Guard:
`gdn_decode.step and recur` at every block.

### GLM's KDA layers came out 1.8% short: an epsilon on the mean
Defect: every KDA layer's output was ~1.8% small on GLM FP4. Cause: `rms_norm(x, 1e-6)` puts epsilon on the mean of
squares; fla's `l2norm` puts it on the sum, and GLM's keys have a mean square near 3e-5. Found by feeding the engine's
own layer inputs to transformers' modules. Fix: `ModelConfig.gdnQkNormEps` (GLM 1e-6/dk).

### NVIDIA's Flash Next FP4 pack prefilled 14% slower: off the tuned kernels
Defect: 4600 vs 5390 tok/s for our pack. Cause: the sorted NAX expert kernel took affine and MXFP4 only, the
dequant+GEMM route affine only. Fix: `Fp4Q` decodes e4m3 scales at group 16; `prefillDqGemm` serves MXFP8: 5340 tok/s.
Trap: stubbing a block changes the router's input and the expert load, so ablated costs read double.

### DSV4.1's tiled indexer scored differently run to run on M1
Defect: a test failed with a different wrong value every CI run on M1. Cause: `simdgroup_barrier(mem_threadgroup)`
does not order per-simdgroup scratch on M1. Fix: `threadgroup_barrier` in that uniform loop.

### Flint prefilled slower on wide chunks: an array-masked SDPA walks every key block
Defect: `--prefill-chunk 2048` beat the machine's 8192. Cause: 35 sliding layers ran MLX's SDPA with an array mask,
which computes every key block. Fix: `nax_attention` takes q/k 128 and serves Flint's sliding prefill with sinks.

### An A/B arm that silently ran the other arm's config
Defect: a "composed" arm measured as fast as the fast path. Cause: `env $VAR "$BIN"` in zsh passes an unquoted `$VAR`
as one malformed assignment. Fix: literal assignments; every lever asserts its engagement line on its own boot and its
absence under the kill switch.

### A CPU caller cached the sigmoid table every GPU kernel reads
Defect: tapes recorded in isolation failed in a suite that ran TTS first. Cause: `swigluSigTable` was built once on
its first caller's stream, the TTS CPU synth. Fix: one table per device. Guard: `the SwiGLU sigmoid table is MLX's
sigmoid on the device that asks for it`.

### A recurrent proof rolled back a full accept and read a row nothing writes
Defect: `plugin recurrent: captured verifies…` failed only when the qmm kernel contract ran earlier in the same
process. Cause: the proof accepted every draft and still rolled back from the capture, reading the state sequence's
last row, which the GDN verify kernels never write (a round rolls back only on `accepted < drafts`); both lanes read
unwritten memory, equal while the pool held zeros, different once the contract's random fill was reused. Fix: the
step is a real partial accept. Guard: the proof itself, now order-independent.

### A threadlocal the host sets was empty in every plugin unit
Defect: under `-Dunits`, a 3D job's CPU stages neither yielded to chat nor sent keepalives, and Flash Next's M-RoPE
path saw no engine context. Cause: each unit compiled its own copy of `gen_yield`, `stream_keepalive` and
`engine_ctx`; `unit_globals.py` read only data/bss sections. Fix: `SharedThreadLocal` (host storage, the current
thread's slot per access); the guard reads `__thread_vars`. Guard: `zig build units-check -Dunits=true`.

### An SSD-first commit freed another entry's arrays (SIGABRT, #947)
Defect: a server with `--prefix-cache-disk` aborted ("pointer being freed was not allocated") in the prefill
write-through and, another day, in `HotPrefixCache.deinit`. Cause: the commit picked its target entry by index, then
`refreshDiskBudget` evicted by swap-remove when free space put the budget under the cap; the stale index freed live
arrays or filed rows under another prompt. Fix: the target crosses the refresh by id (`indexOfId`); an evicted target is
a fresh store. Guard: the two `across an SSD-first refresh that evicts` tests, each in both orders.

### A plugin that rewrote `Forward.tokens` heard nothing in the live engine (nemo_speechlm)
Defect: Nemotron Transcribe answered "We are." to a spoken sentence; unit tests were green. Cause: its `forward` swapped
the out-of-table `<|audio|>` id for a stand-in in `f.tokens`, but the engine sends ids as a device array
(`Forward.ids`) and leaves `tokens` empty, so the swap never ran, the splice never matched and the rows were dropped.
Fix: the swap covers both forms. Rule: a plugin that reads or rewrites ids handles `f.ids` as well as `f.tokens`.
Guard: the splice proof runs the same prefill through host tokens and through a device array and demands equal logits.

### Native slot switches shared Engram prompt and confidence policy
DeepSeek-V4.1 parked its State while its Engram prompt, read-ahead and confidence demand stayed on the model.
Switching back could read another request's prompt and policy. These fields now follow the State; parking drains
read-ahead after reserving map capacity, and arming a fresh prompt clears confidence demand. Guard: the
`deepseek_v41 slot request` regressions cover resident auxiliary-cache and legacy streamed configurations.

### An embeddings request killed a DeepSeek-V4.1 server
Defect: `POST /v1/embeddings` against a loaded DeepSeek-V4.1 logged `[mlx] expected a non-empty mlx_array` twice, then
segfaulted the server. Cause: the embedding forward asks the plugin for hidden rows only (`want_hidden`, no logits);
DeepSeek-V4/V4.1's `forward` ignored the flag, ran a position-0 prompt pass that reset the live conversation state,
returned `.ok` with no hidden rows, and the driver handed the empty array to the pooling. Fix: both plugins refuse
`want_hidden` before any forward; the driver fails a hidden-only forward that returns none; the embeddings handler
refuses a module-owned-state model (`Caps.owns_decode_state`) with a named 400. Guard: `tests/test_dsv41.sh` [7a]
(refused by name, the next greedy request byte-equal to solo, no `[mlx]` line).

### A cancelled DeepSeek-V4.1 request threw away the conversation
Defect: stop a DeepSeek-V4.1 reply, send the next message, and without `--prefix-cache-disk` the whole conversation
prefilled again. Cause: a RAM restore of a module checkpoint removed the entry and handed its buffers to the slot; a
cancelled request commits nothing, so the state died with the slot. Fix: the restore shares the entry (`restoreModule`:
MLX copies a shared buffer on the slot's first write, and the prefill bill already prices the state whole); the KV
checkout refuses module entries. Guard: `HotPrefixCache module: a restore shares the checkpoint, so a request that never
commits leaves it resumable`.

### DeepSeek-V4.1's DSpark rounds verified almost nothing their head was sure of
Defect: DSpark group rounds verified no draft in a quarter of rounds while verified drafts landed 96-100% a position.
Cause: the round policy's calibration started from a logistic prior fitted on GLM-5.3's DFlash drafter; DSpark's
confidence head is itself calibrated (1,535 full-block verifies: accepted ≈ sigmoid of the logit, every position), and
the prior read its 0.93 as 0.65, under the group's row threshold, so those bins never saw a verify to learn from. Fix:
DSpark's calibration starts from the head (`Calibration.head`); the recorded tapes pin the old prior. Guard:
`Calibration: a head prior reads the drafter's confidence as its acceptance until verifies say otherwise`.

### A slot in a DSpark group round stopped drafting because of its partner
Defect: two DeepSeek-V4.1 coding streams; the second slot skipped drafting in 143 of 162 rounds although its drafts
landed 86-100% a position. Cause: each slot's drafting backoff weighed its tokens against the whole round's wall,
which holds the other slot's verify rows. Fix: a slot is charged its share of the stage pass and its own rows, against
its accepted drafts at the group's plain cost a token (`moduleGroupRound`). Guard: `tests/test_dsv41.sh` [8b]
(`group_rounds=` engaged); read `plain_ticks=` on a two-stream code run when the backoff changes.
