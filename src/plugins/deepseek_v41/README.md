# DeepSeek-V4.1-Flash-1017 NVFP4 in mlx-serve

Status: production guide for the `deepseek_v41` runner in mlx-serve core, release head `a99fc4f6` (PR 1256),
2026-11-24. "We" are the PR 1256 authors, who tuned and measured on M5 Ultra Macs with 512 GB of memory. The PR body
holds the A/B evidence. Agent and code notes for this folder are in [AGENTS.md](AGENTS.md).

## Summary

This plugin serves NVIDIA's `DeepSeek-V4.1-Flash-1017-NVFP4` checkpoint on Apple GPUs, with the full
1,048,576-token context. A converter rewrites only the checkpoint's file headers, so the weights keep NVIDIA's bytes
and the routed experts stay in NVFP4. DeepSeek's own drafter, DSpark, proposes tokens that the model checks in one
step, and this makes decode faster. The defaults are the release configuration, and you set no switch to get them.
On an M5 Ultra with 512 GB, the control build of this release decoded 97.43 tok/s greedy and prefilled 5,549 tok/s on
our standard test. Against the pack as shipped, the defaults on M5 have one known loss: an 8-bit LM head, whose
KLD measured equal to the bf16 head's. Macs without neural accelerators also serve a lossy f16 attention core by
default. Every other default-on switch is exact, a reordered sum no farther from float64, a change to the drafts only, or MLX's TF32 matmul. HumanEval,
the long-context sweep and the final numbers on the release head are not done yet.

## Serve the model on an M5 Ultra with 512 GB

Do these steps in this order.

1. Convert the checkpoint on the machine that holds it:
   `python3 -I scripts/dsv41_nvfp4_pack.py SRC_DIR DST_DIR --check`.
2. Put a `draft-vocab.txt` file in the pack directory (see "Speculative decoding"). Without it, decode is slower.
3. Raise Metal's wired-memory limit to 512000 MB (500 GiB): `sudo sysctl iogpu.wired_limit_mb=512000`.
4. Start the server: `mlx-serve --model <pack> --serve --warmup-full`.

`sysctl` sets the wired limit for the running system only, so set it again after each restart. Every run on this
release's stack used 512000. Metal's working-set limit can refuse a load that RAM holds. The engine then stops the load,
names `iogpu.wired_limit_mb` in the error, and logs the `sudo sysctl` command that admits the load. That value is the
load's need (the weights plus warm-up headroom) plus an 8 GiB margin. We have not tried this release at macOS's default
limit.

Metal compiles each kernel at its first use. With `--warmup-full`, the server sends itself a ladder of prompt sizes
before it reports ready, so no user request waits for a first compile. This takes from seconds to a minute, on a
build's first load of the model only.

## What the pack holds

The converter `scripts/dsv41_nvfp4_pack.py` makes each shard an APFS clone and rewrites its header at the same length,
so it copies no tensor data. `--check` reads the result again and compares each converted tensor with the source.

- The fp8 linears become MLX mxfp8 at group 32. A side shard holds their row scales.
- The DSpark stages' routed experts become MLX mxfp4 at group 32.
- The Engram tables (DeepSeek-V4.1's hashed n-gram embeddings) stay mxfp4, as stored.
- The routed experts stay NVFP4 (MLX nvfp4 at group 16). ModelOpt's per-expert global scales move to a second side
  shard, `model-expert-scales.safetensors`. The gate scale applies before the SiLU, the up scale before the SwiGLU
  clamp, and the down scale on the routing weight.

NVFP4 is the format NVIDIA ships. Its 16-value groups and e4m3 scales hold spiky weights better than MXFP4, so in
general it is the more accurate format. `--experts-mxfp4` writes the experts as MXFP4 instead. That form is
value-identical to NVIDIA's for this checkpoint only, because all of this checkpoint's scales are powers of two; the
converter checked 16,986,931,200 groups. In a comparison before this release's speed changes, the MXFP4 form prefilled
faster (4,108 against 3,994 tok/s) and was 18 GB smaller. The plugin reads either form from the pack, with no flag.

The pack ships the LM head in bf16. At load, the plugin requantizes it to 8-bit affine at group 64
(`MLX_SERVE_DSV41_HEAD8`, on). The trunk and the drafter share this head. This is the one quantization change in the
defaults. On 50 prompts and 1,600 positions, the 8-bit head's mean KLD was 0.1020 and the bf16 head's was 0.1021. The
paired difference was +0.00007 ± 0.00025 (one standard error), and top-1 agreement was 92.6% for both. The reference
is the top-20 log-probabilities that DeepSeek's API returned on 2026-09-10 for `deepseek-flash` (the ds4 repo's
`deepseek-v4.1-flash-20260910-general` set, weights revision `df42c109`). That model can differ from the 1017
checkpoint, so the absolute KLD includes the difference between the two models; the paired difference measures the
head. The bf16 head (`=0`) cost 4.95% of spec-on decode speed and used 0.63 GB more memory.

## Speculative decoding (DSpark)

The pack ships DeepSeek's DSpark stages, a small drafter that proposes a block of tokens. The model then checks the
block in one wide step, the verify, which accepts or replaces every proposed token. The plugin loads the stages
whenever the pack ships them. `--no-mtp` keeps them off, and `MLX_SERVE_DSV4_DSPARK=0` turns drafting off. A policy
picks how many rows each verify checks (`MLX_SERVE_DSPARK_POLICY2`).

For a greedy request, the drafter proposes its argmax tokens, and the verify keeps the longest prefix that the model
agrees with. The verify rows run multi-row kernels, which do the math of a plain decode step in another sum order. So
greedy text with DSpark on can differ from plain decode by rounding. `MLX_SERVE_DSV41_EXACT_VERIFY=1` makes each verify
row run its plain step's kernels, and greedy DSpark text is then plain decode's bytes, at some speed cost.

For a sampled request (`MLX_SERVE_DSV41_DRAFT_SAMPLE`, on), the drafter draws each token from its own distribution q.
The verify keeps it with probability min(1, p / q), where p is the model's distribution, and resamples a rejected token
from max(0, p - q). This is standard speculative sampling, so the output keeps the model's distribution.

The draft vocabulary (`MLX_SERVE_DSV41_DRAFT_VOCAB`) is a `draft-vocab.txt` file in the pack directory. It holds
decimal token ids separated by whitespace; our runs used 65,536 ids. Greedy drafts then score only those ids, plus up
to 16,384 of the request's own most frequent prompt ids (`DRAFT_VOCAB_PROMPT`, on). The trunk and every verify keep the
whole vocabulary, so only the drafts change. A sampled request always drafts over the whole vocabulary.
`MLX_SERVE_DSV41_DRAFT_VOCAB=<path>` names another file, and an empty value or `0` turns the vocabulary off. The file
gave +1.46% decode on the standard test. Neither the converter nor any script in this repo writes it.

## The defaults on this Mac and on others

Each switch is an environment variable `MLX_SERVE_DSV41_<NAME>`, registered in `src/env.zig` with its default and one
line of doc. A default-on switch turns off with `=0`. Each speed change in this release won its A/B on an M5 Ultra with
512 GB, except three washes kept because they remove work.

Other Macs get the same defaults. Some parts scale or turn off by themselves:

- A Mac that cannot hold the pack refuses the load. With 256 GB or less, the routed experts must stream from disk
  (the per-model `stream_experts` setting), and disk reads then limit decode.
- The prompt pass width scales with memory (`PF_PASS_SIZE`).
- Where the Engram tables do not fit in memory, the plugin reads them from disk at each lookup. The switches that need
  every table in memory then turn off, and the kernel family falls back to the `ssd` path (`WEIGHT_PATH`).
- GPUs without neural accelerators (NAX, the M5 GPU's matrix units) do not run the matrix-unit kernels. This covers
  M4 and older, and an M5 on macOS before 26.2.

We did not tune or measure any machine other than the M5 Ultra with 512 GB.

## Memory and the 1M-token context

The plugin bills memory in four parts, and the engine admits a load or a request only when the sum fits: the resident
load, the compressed KV cache, the window rings, and the prompt pass. The KV cache costs only 3,200 bytes a token
(3.36 GB at 1,048,576 tokens), because only four layers store KV.

On an earlier build of this release with spec decode off, a 1,038,931-token request at a 1,048,576-token context was
admitted, with 40,230 MB to spare. That build did not have the 4 GB Engram partials' cache, so by arithmetic the margin
is now about 36 GB. On the control build, a load at a 1,064,960-token context succeeded and admitted a short request.
A real 1M-token prompt has not run on the release head yet.

<details>
<summary>The four parts of the memory bill</summary>

| Part | Where the code computes it | Size for this model |
|---|---|---|
| Resident load | `residentBytes` (`plugin.zig`) calls `residentDiskBytes` (`deepseek_v41.zig`) | The pack's shards less the tensors it drops, plus the Engram tables it holds, the Engram LRU (256 MiB), the Engram partials' cache (4 GB) and the draft-only head |
| Compressed KV per token | `kvPerToken` (`bill.zig`) | 3,200 bytes a token |
| Window rings | the rings part of `prefillBytes`'s `state` term (`deepseek_v41.zig`) | 40 layers x (128 + 16) rows x 512 x 2 bytes, about 5.9 MB |
| Prompt pass | `prefillBytes` (`deepseek_v41.zig`), plus the `PF_KEEP_WIDE` reserve | Grows with the pass width. `PF_PASS_SIZE` picks the widest pass that fits the memory share; a set `--prefill-chunk` narrows it |

Layers 2, 8, 14 and 20 are the KV sources. Each stores a 512-wide latent and a 128-wide index key per token in bf16,
compressed 2:1 in the first three and 1:1 in layer 20.

The 1M admission ran on build `c59b60c6`. The admission check needed 38,246 MB and had 78,476 MB available, with a
20.03 GB prompt reserve armed. The partials' cache bills 4 GB and allocates 3,787 MiB on this pack. The 1,064,960-token
load ran on control `04b8f7d0`, with a 17,769 MB reserve at 44,544-row passes, and admitted a 4-token request.

`PF_KEEP_WIDE`'s reserve needs `patches/mlx-metal-wide-reserve.patch`, which adds a reserve of large Metal buffers to
MLX's allocator. A prompt pass then does not map fresh pages in every request. Without the patched libmlx, the plugin
declines the switch by name and logs why.

</details>

## Accuracy: what "lossless" means here

"Lossless" means no change of precision, quantization, routing or approximation against the pack as shipped. A
reordered sum or a changed tie-break is allowed when the result is no farther from float64. Each default-on switch is
in one of four classes, with two exceptions.

| Class | What it means | Default-on switches |
|---|---|---|
| EXACT | Bit-identical to the `=0` path | 30 |
| REORDER | The same math in another sum order or tie-break. Over a test set, the total l2 error and the worst element against float64 are no farther than the `=0` path's. Two switches are held to stated bounds instead (see the results below) | 14 |
| DRAFT-ONLY | Changes only the drafts. The verify accepts or replaces every draft | 5, plus `DRAFT_VOCAB` |
| TF32 | MLX's TF32 matmul (`MLX_ENABLE_TF32`, on). The verified path is exact; only the drafter's P·V rounds | 1 (engine) |

The first exception is `HEAD8`, the 8-bit LM head described in "What the pack holds".

The second exception is `ATTN_F16`, an attention core with f16 scores and probabilities. It is lossy. On M5, at the
release's prompt spans, its bf16-out l2 against float64 was 1.80e-3 to 2.31e-3, against 1.63e-3 to 1.68e-3 for the f32
chain (`=0`). That is 10% to 37% farther.

One function, `attnCore`, picks the core for every prompt-attention call. It decides from the call's shape, dtypes and
device and the load's switches only, never from the caller. On NAX it takes the first core that fits, in this order:

1. the flash kernel (`FLASH_ATTN`), at any pick count;
2. the gather, where the pick count (the compressed rows that the indexer picks) is a multiple of 64;
3. the f32 chain.

The f16 core serves only on Macs without NAX, where it is the fast path and on by default as upstream set it, or under
an explicit `ATTN_F16=1`. On those Macs `=0` gives the f32 chain, and we have not measured them. The load fixes the
switch's effective value, and the prompt cache's layout hash includes it. On M5 the flash kernel is nearer to float64
than the f32 chain: at the release's prompt spans, bf16-out l2 1.660e-3 to 1.670e-3 against 1.667e-3 to 1.677e-3, and
f32-out l2 6.2e-5 to 6.6e-5 against 1.52e-4 to 1.67e-4.

A tensor split (32 heads a rank) takes the f32 chain. On the f32 chain's routes (a tensor split, `FLASH_ATTN=0` at a
pick count that is not a multiple of 64, and f16 or f32 activations), attention's output depends on the chunk width.
MLX picks its GEMM by row count, so the sums run in another order: a reorder. Flash and the gather do not depend on it.

Two fixes in a row each left a path to the f16 core on M5, because each caller chose its own core. The root-cause fix
gives the choice one owner, `attnCore`. Tests drive the served routes through the plugin's entry points on a
full-model fixture and count each core's calls; on NAX they count 0 f16 calls.

The class tests are in the plugin. Each test also runs a planted fault as a control, and the control must fail.
`prompt/wide_test.zig` holds the prompt switches. `decode/attn.zig`, `decode/woa_plain.zig` and
`decode/engram_ids.zig` hold `ATTN_DEC`, `WOA_PLAIN` and `ENGRAM_GPU_IDS` against float64. `tf32_test.zig` runs the
served f32 matmuls against float64, over operands saved from a served run (`DSV41_TF32_FIXTURE`).

`FP8_FOLD` and `FP4_READS` read three rare scale codes differently from MLX. At load, the plugin scans each layer for
those codes and turns both switches off, by name, where it finds one. The shipped NVFP4 pack holds none of them.

<details>
<summary>Results of the word and float64 tests</summary>

- `MOE_TAIL` (EXACT): one launch gives the two launches' words, with 0 differing words at 17 to 8,192 rows.
- `FLASH_ATTN` (REORDER): its f32 output is nearer to float64 than the gather's, 6.4e-5 against 1.8e-4 relative.
- `MOE_PAIR_SCALED` (REORDER, bounded): at 384 experts, 0.223% to 0.237% of the activation words differ from the split
  path. The whole-sample float64 l2 is 1.0000 times the split path's. The test requires each differing word within one
  bf16 step of its row's largest value, under 1% of words differing, and a sample l2 within 1.05 times the split
  path's. The words more than one step apart in bf16 code are words that the split path gives as exactly 0.0, where
  the paired kernel keeps a tiny value, about 1e-35.
- `PF_HC_V2` (REORDER, bounded): an f32 reorder of the five-pass hyper-connections. Its f32 mixes and split are 2 to 6
  times farther from float64 than the five passes', at the 1e-7 level. That is below the bf16 rounding that follows,
  and the tail's output words are the five passes' words. The test holds the mixes and split within 1e-5 of float64,
  and every other output within 1.01 times the five passes' l2.
- `ATTN_DEC` (REORDER): over 72 cells, nearer to float64 than `dec_attn` (`=0`) in total l2 and worst element. Its
  window-only keys run 2 splits of 64: l2 / max 1.599e-7 / 4.056e-7 against 1.632e-7 / 4.439e-7 with the float64 merge,
  and 1.688e-7 / 4.141e-7 against 1.769e-7 / 4.634e-7 with the f32 merge.
- `WOA_PLAIN` (REORDER): nearer to float64 than its `=0` path in total l2 and worst element.
- `ENGRAM_GPU_IDS` (REORDER): 30 of 409,600 bf16 words differ from MLX's chain (25 nearer, 4 farther). The worst
  element is equal, and the total squared error is larger by a factor of 1 + 4.5e-9.
- TF32: the router, the compressor projection and the DSpark q·k read 0 operand elements that TF32 rounds. The hc
  collapse has operands that TF32 would round, but MLX serves it on its gemv kernel, which does not use TF32, so its
  error is the same with TF32 on and off. The only TF32 loss is the drafter's P·V, at a relative l2 of 2.8e-4 to
  2.9e-4, and it changes only the drafts.

</details>

<details>
<summary>Every default-on switch and its class</summary>

Names omit `MLX_SERVE_DSV41_`. "Evidence" names the evidence kind: a standing test in the plugin, the kernel contract,
or the text of the A/B and the commit that added the switch.

| Switch | What it does | Class | Evidence |
|---|---|---|---|
| PROMPT_ASYNC_EVAL | Commits the last wide prompt layer without a wait | EXACT | commit |
| PF_CED_HC | CED compressor hc pre-norm once over the pass | EXACT | A/B text |
| PF_ROPE_PAIRS | Query-tail rope computes each angle once a pair | EXACT | A/B text |
| PF_RMS_ROWS | Query-latent rms reads its rows in place | EXACT | A/B text |
| CED_LAZY | Runs a CED replay once, for its first reader | EXACT | commit |
| IN_PROJ_MIXR | Verify in_proj hc mixes as one threadgroup a (mix, row) | EXACT | A/B text |
| HC_EXPAND_NORM | FFN-side hc expand and norm in one launch | EXACT | commit |
| PF_MOE_EPI | Counting-sort route pack in place of two argsorts | EXACT | test |
| PF_GU_SEG | 128 x 128 paired gate/up where the planner picks it | EXACT | A/B text |
| PF_KV_ROWS | Window kv rows on one simdgroup a row | EXACT | A/B text |
| PF_DOWN_TOKORD | Staged down writes rows in token order | EXACT | test |
| PF_FP4_GEMM | Staged matrix-unit gather for routed gate/up and down | EXACT | test |
| WOA_JOIN | Joins the spans' wo_a rows in one copy | EXACT | commit |
| PF_ROUTE_PACK | Router top-6 on a simdgroup a row; one-threadgroup scan | EXACT | test |
| PF_ROUTER_BF16 | Router logits on a bf16 MMA, only where MLX runs one sum (1,707 rows and more) | EXACT | test |
| PF_KEEP_WIDE | Prompt buffers from the MLX patch's reserve | EXACT (memory) | test |
| PF_ENGRAM_PASS | Engram wkv once per 16 spans | EXACT | test |
| MOE_TAIL | MoE combine and hc expand in one launch | EXACT | test |
| FLASH_TRIM | FLASH_ATTN with a raw-score block max and 8-byte stores | EXACT (to FLASH_ATTN) | test |
| FFN_IN_SINK | ffn_in mix tail in its own launch | EXACT | A/B text |
| SPEC_ENGRAM_AHEAD | Queues the next verify's Engram kv behind the draft | EXACT | test |
| FFN_IN_R5_COLS1 | 5-row ffn_in launch 1 with one column a simdgroup | EXACT | A/B text |
| DRAFT_POS_ONCE | Makes draft ring slots and positions once | EXACT | commit |
| FUSE_WQB_RING | The wq_b launch writes the window ring | EXACT | contract |
| ENGRAM_KV_CACHE | GPU cache of Engram slot partials, at most 4 GB | EXACT (to ENGRAM_KV_SLOTS) | test |
| FP4_DEC_WORD | Decodes e2m1 codes a word at a time | EXACT | test |
| FP4_UP_SPLIT | Shared gate and up rows on separate simdgroups | EXACT | test |
| FP8_FOLD | Folds 2^8 into the e8m0 scale in mxfp8 reads | EXACT (declines on rare codes) | commit, load scan |
| MOE_DOWN_ORDER | Verify down launch in column-block order | EXACT | A/B text |
| FUSE_DOWN_SCALE | NVFP4 down scale inside the gate/up launch | EXACT | commit |
| CED | Prefill decoder layers over the last window rows only | REORDER (row count) | by construction |
| MOE_FP4_NAX | Fused fp4 MoE decode kernels on the matrix units | REORDER | test |
| MOE_PAIR_SCALED | NVFP4 prompt gate/up, scales and activation in one launch | REORDER (bounded) | test |
| PF_HC_V2 | Wide-layer hc in three passes, not five | REORDER (bounded) | test |
| PF_PASS_SIZE | Near-equal memory-sized prompt passes | REORDER (partition) | commit |
| PF_IDX_NAX | Indexer scores on the matrix units | REORDER | commit |
| PF_COMP_BF16 | Compressor projection on a bf16 MMA, K blocks summed apart | REORDER | test |
| FLASH_ATTN | Every prompt-attention call on one flash kernel where it fits | REORDER | test |
| ENGRAM_GPU_IDS | Engram ids hashed on the GPU; fused wkv | REORDER | test |
| ENGRAM_KV_SLOTS | Slot-blocked Engram wkv | REORDER | test |
| FP4_READS | Bit-placement fp4 reads; each 16-group summed apart | REORDER (declines on rare codes) | test |
| ATTN_DEC | Decode attention on a shape-cut tile kernel | REORDER | test |
| WOA_PLAIN | Splits merged once; wo_a as a plain grouped GEMV | REORDER | test |
| FUSE_MXFP8_QMM | Prompt mxfp8 linears on MLX's quantized matmul | REORDER | test |
| DRAFT_FUSED | DSpark stages on the fused decode kernels | DRAFT-ONLY | commit |
| DRAFT_SAMPLE | Sampled drafts from q, kept at min(1, p / q) | DRAFT-ONLY | test |
| DRAFT_QKV | Fuses a stage's wq_a and wkv at load | DRAFT-ONLY | A/B |
| SPEC_SYNC | Greedy accept on the GPU; next draft queued behind the verify | DRAFT-ONLY | A/B text |
| DRAFT_VOCAB_PROMPT | A request's prompt ids join the draft vocabulary | DRAFT-ONLY | test |
| HEAD8 | LM head requantized to 8-bit g64 | Requantized (KLD equal) | KLD measurement |
| ATTN_F16 | f16 attention core; on by default only without NAX | LOSSY (measured, not gated) | printed float64 measurement, served-route tests |

Engine variables on the DeepSeek path: `MLX_SERVE_DSV4_DSPARK` (REORDER: the verify's multi-row kernels),
`MLX_SERVE_DSPARK_POLICY2` and `MLX_SERVE_DSPARK_POLICY2_MODELED` (DRAFT-ONLY: they pick verify widths) and
`MLX_ENABLE_TF32` (TF32).

</details>

## Switches that stay opt-in

Six switches stay off by default, and three value switches set memory and kernel choices.

<details>
<summary>The opt-in and value switches</summary>

Each of these is off by default. Names omit `MLX_SERVE_DSV41_` except where shown.

- `EXACT_VERIFY=1` (EXACT) makes greedy DSpark text equal plain decode's bytes on M5.
- `EXACT=1|solo` (EXACT) does the same on M3-generation or later Apple GPUs without NAX. Any value other than unset,
  empty or `0` turns it on. On a GPU with NAX, the load leaves it off and logs why.
- `FLASH_PSUM=1` (REORDER) is a flash-attention variant that is nearer to float64. It measured no faster.
- `MXFP4_VIEW=1` (REORDER) loads an NVFP4 bank whose every 32-value span is one power of two as MXFP4.
- `MOE_PAIR=1` (REORDER) is the paired prompt gate/up for MXFP4 banks without per-expert scales.
- `MLX_SERVE_DSPARK_PIN_ROWS=N` (for tests) makes every DSpark round verify N rows, for width sweeps.

Three value switches set memory and kernel choices. `ENGRAM_LRU_MB` (256) sets memory only. `ENGRAM_RESIDENT` (`auto`)
and `WEIGHT_PATH` (`auto`) can turn off the resident fusions. The `ssd` path turns off `FUSE_MXFP8_QMM`. Tables not
held in memory turn off `ENGRAM_GPU_IDS` and the Engram KV cache.

</details>

## Results

The standard test is three coding prompts of about 17,600 tokens each, with 1,024 greedy completion tokens and a fresh
server per run. On control `04b8f7d0` it gave these speeds:

| Cell | Prefill tok/s | Decode tok/s |
|---|---|---|
| Greedy, DSpark on | 5,548.7 | 97.43 |
| The model card's sampling settings, DSpark on | 5,535.3 | 88.06 |
| Greedy, DSpark off | 5,534.1 | 61.48 |
| Prompt p10, greedy, DSpark on | 5,530.0 | 106.45 |

The control is the stack before the upstream merge and the release fixes. The release's attention fix sends the CED
replay of a long prompt to the flash kernel, so greedy texts on the release head differ from the control's. The PR
body holds every run and the A/B of each switch. The final numbers on the release head are not done yet.

## Gaps

| Gap | What the release needs |
|---|---|
| HumanEval, the 1K to 1M context sweep and the final standard test have not run on the release head | Those runs, posted in the PR |
| Without NAX, the lossy f16 attention core (`ATTN_F16`) is on by default, and we have not measured it on those Macs | A float64 test on those Macs, or `ATTN_F16` off there |
| On the f32 attention chain's routes, attention's output depends on the chunk width | A test that holds each chunk width no farther from float64 |
| `HEAD8` is a quantization change. Its KLD equals the bf16 head's on our set, but it is not bit-identical | A ruling to keep it, or `=0` by default |
| No tool in this repo makes `draft-vocab.txt` | The release ships the file with the pack |
| We tuned and measured only on M5 Ultra Macs with 512 GB | Runs on smaller Macs and older GPUs |
| A real 1M-token prompt has not run on the release head, and the release has not loaded at the default wired limit | The 1M point of the sweep; one load at the default limit |
| We have no PPL or KLD of the whole stack; the KLD above is of the head alone | A whole-stack KLD against the same reference |

## Terms

| Term | Meaning |
|---|---|
| NVFP4 | NVIDIA's 4-bit float format: e2m1 values, one e4m3 scale per 16, plus a per-tensor global scale |
| MXFP4, mxfp8 | The OCP microscaling formats: one e8m0 scale per 32 values |
| DSpark | DeepSeek-V4.1's own drafter stages, used for speculative decoding |
| Verify | One step of the model over the current token and the drafts |
| Engram | DeepSeek-V4.1's hashed n-gram embedding tables |
| hc | DeepSeek-V4.1's hyper-connections, the multi-stream residual |
| CED | The plugin's chunked prompt path, which replays chunks through the decoder layers |
| NAX | The M5 GPU's neural accelerators (matrix units) |
| Span | The 256-row unit that a wide prompt layer's attention takes |
| Pick count | The number of compressed rows that the indexer picks for a span's attention |
| KLD | Kullback-Leibler divergence of the served distribution from a reference |
| `=0` path | The path a default-on switch serves when it is set to 0: the path before the change |
| Standard test | Three coding prompts of about 17,600 tokens, 1,024 greedy tokens, a fresh server per run |
