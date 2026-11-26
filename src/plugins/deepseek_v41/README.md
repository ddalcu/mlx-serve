# DeepSeek-V4.1-Flash-1017 NVFP4 in mlx-serve

Status: production guide for the `deepseek_v41` runner in mlx-serve core, PR 1256, 2026-11-24. We measured build
`a99fc4f6`. Six later commits change product or test code paths, and the defaults' kernels compute the same values (the PR's
reply comment 4 lists them and their checks). "We" are the PR 1256 authors, who tuned and measured on
M5 Ultra Macs with 512 GiB (549.8 GB) of memory; GB here means 1e9 bytes. The PR's reply comment 1 holds every A/B. Agent and code
notes for this folder are in [AGENTS.md](AGENTS.md).

## Summary

This plugin serves NVIDIA's `DeepSeek-V4.1-Flash-1017-NVFP4` checkpoint on Apple GPUs, up to the full
1,048,576-token context. A converter rewrites the checkpoint's file headers and adds two side shards for scales. The
weight codes keep NVIDIA's bytes, and the routed experts stay in NVFP4. DeepSeek's own drafter, DSpark, proposes
tokens that the model checks in one step, which makes decode faster. The defaults are the release configuration.

On an M5 Ultra, the release build decoded 98.80 tok/s greedy with DSpark on and prefilled 5,551 tok/s on the standard
test's p01 prompt. The first NVFP4 build gave 71.77 and 3,994 tok/s (see "Results"). Both runs used the NVFP4 pack, so
the pack gives none of the gain; the runs also differ in setup.

The plugin has 56 default-on switches (55 on M5, where `ATTN_F16` is off). We count 54 of them as lossless.
"Lossless" here means each switch's output is bit-identical to its `=0` path (EXACT); or (REORDER) not
bit-identical but measured no farther from float64 (6 switches), held to a stated bound that can allow somewhat more
error than `=0` (7), or classed by reasoning or A/B text alone (3); or changes only drafts and verify widths, or how a sampled verify accepts drafts while keeping the model's
distribution
(DRAFT-ONLY). `MLX_ENABLE_TF32` has its own class (TF32): on M5 at the defaults it rounds only the drafter's P·V.
Greedy text can still change. No test compares the whole model with a reference implementation (see "Accuracy"). The two outside are an 8-bit LM
head, whose KLD was within one standard error of the bf16 head's, and a lossy f16 attention core. That core is on by
default only without NAX. NAX is the M5 GPU's matrix units; M4 and older Macs have none. On EvalPlus the release
build scored 161/164 on HumanEval and 366/378 on MBPP (base tests).

## Serve the model on an M5 Ultra with 512 GiB

1. Build the server from this branch, as "Server only" in [docs/building.md](../../../docs/building.md) says. Its
   `./scripts/build-mlx.sh` step also applies the MLX patch that this release needs.
2. Download NVIDIA's checkpoint from https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 to SRC_DIR.
3. Convert it on the same machine: `python3 -I scripts/dsv41_nvfp4_pack.py SRC_DIR DST_DIR`. The converter checks
   the result when it finishes. Put DST_DIR on the same APFS volume as SRC_DIR. Keep SRC_DIR, because a shard with
   no change is a symlink to it. The converter also writes the draft vocabulary, `DST_DIR/draft-vocab.txt` (see
   "The draft vocabulary" below).
4. Raise Metal's wired-memory limit to 512000 MiB (500 GiB, 536.9 GB): `sudo sysctl iogpu.wired_limit_mb=512000`.
5. Start the server: `./zig-out/bin/mlx-serve --model DST_DIR --serve --warmup-full`. It listens on port 11234 by default.
6. Get the model's id: `curl -s 127.0.0.1:11234/v1/models`.
7. Send a first request:
   `curl -s 127.0.0.1:11234/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "<id>", "messages": [{"role": "user", "content": "Hello"}]}'`.

The loaded model needs about 433 GB, and a 17,600-token request peaks at about 452 GB (at `--ctx-size 32768`; at the
default 1M context the prompt reserve is 4.9 GB larger, and we have not measured it), so a Mac needs 512 GiB to hold
the NVFP4 pack. `sysctl` sets the wired limit for the running system only, so set it again after each restart.
`--warmup-full` compiles the Metal kernels before the server reports ready.

<details>
<summary>The wired limit, memory and warm-up, in detail</summary>

To check a converted pack again later, run the converter with `--check`; it then checks and does not convert.

Every run in this
release's A/Bs and cells used 512000 MiB; the first NVFP4 build's A/B used 440320 MiB. If Metal's working-set limit refuses a load, the engine's error names the
`sudo sysctl` command that admits it. We have not tried this release at macOS's default limit. The build that PR 1256
showed before this release reported one load at the default limit; its record names no build.

The 433 GB is the idle load of build `c59b60c6` (429 GB), measured before the 4 GB Engram partials cache, plus that cache. On the standard test the release build peaked at 452.3 GB (three-prompt mean; 452.9 GB at most, on p14).

Metal compiles each kernel at its first use. With `--warmup-full`, the server sends itself a ladder of prompt sizes
before it reports ready, so no user request waits for a first compile. This takes from seconds to a minute, on a
build's first load of the model only.

</details>

## What the pack holds

The converter `scripts/dsv41_nvfp4_pack.py` makes each shard an APFS clone and rewrites its header at the same length,
so it copies no tensor data for the default form. Its check then reads the result again and compares each converted
tensor with the source.

- The fp8 linears become MLX mxfp8 at group 32. A side shard holds their row scales.
- The DSpark stages' routed experts are already MLX mxfp4 at group 32. The converter only renames their scales.
- The Engram tables (DeepSeek-V4.1's hashed n-gram embeddings) stay mxfp4, as stored.
- The routed experts stay NVFP4 (MLX nvfp4 at group 16). The per-expert global scales of ModelOpt (NVIDIA's
  quantizer) move to a second side shard, `model-expert-scales.safetensors`.

NVFP4 is the format NVIDIA ships. Its 16-value groups and e4m3 scales hold spiky weights better than MXFP4, so in
general it is the more accurate format. `--experts-mxfp4` writes the experts as MXFP4 instead (see the block below).

The pack ships the LM head in bf16. At load, the plugin requantizes it to 8-bit affine at group 64
(`MLX_SERVE_DSV41_HEAD8`, on). The trunk and the drafter share this head. Its KLD was within one standard error of the bf16 head's
(the block below has the numbers). The bf16 head (`=0`) cost 4.95% of decode speed with DSpark on.

<details>
<summary>The MXFP4 form, the LM head's KLD and the expert scales</summary>

On 50 prompts and 1,600 positions, the 8-bit head's mean KLD was 0.1020 and the bf16 head's was 0.1021. The paired
difference, bf16 minus 8-bit, was +0.00007 ± 0.00025 (one standard error), and top-1 agreement was 92.6% for both. The
bf16 head's 4.95% cost comes from the unrounded means, and it used 0.63 GB more memory.

`--experts-mxfp4` (it needs numpy) writes the experts as MXFP4 instead. This form writes the expert shards whole, so
it needs disk space for them. That
form is value-identical to NVIDIA's for this checkpoint only, because all of this checkpoint's scales are powers of
two. The converter checked 16,986,931,200 groups. The plugin reads either form from the pack, with no flag.

| Expert form, build `f7e4da9e` (A/B `nvpf-dsproof-ab`, fused fp4 decode off) | Prefill tok/s | Peak GB |
|---|---|---|
| NVFP4 (the default) | 3,994 | 438.6 |
| MXFP4 (`--experts-mxfp4`) | 4,108 | 420.7 |

The KLD reference is the top-20 log-probabilities that DeepSeek's API returned on 2026-09-10 for `deepseek-flash`
(the ds4 repo's `deepseek-v4.1-flash-20260910-general` set, weights revision `df42c109`). That model can differ from the
1017 checkpoint. So the absolute KLD includes the difference between the two models, and the paired difference
measures the head.

The gate's per-expert scale applies before the SiLU, the up scale before the SwiGLU clamp, and the down scale on the
routing weight.

</details>

## Speculative decoding (DSpark)

The pack ships DeepSeek's DSpark stages, a small drafter that proposes a block of tokens. The model then checks the
block in one wide step, the verify, which accepts or replaces every proposed token.
`--no-mtp` keeps them off and frees their memory, unless `MLX_SERVE_DSV4_DSPARK` is set
to a value that does not start with 0; then the stages load, and drafting stays off. `MLX_SERVE_DSV4_DSPARK=0` turns drafting off, but the stages stay loaded. A policy picks how many rows each verify checks
(`MLX_SERVE_DSPARK_POLICY2`).

For a greedy request, the drafter proposes its argmax tokens, and the verify keeps the longest prefix that the model
agrees with. The verify rows run multi-row kernels, which do the math of a plain decode step in another sum order. So
greedy text with DSpark on can differ from plain decode by rounding. Under `MLX_SERVE_DSV41_EXACT=1`, greedy DSpark
text is plain decode's bytes, one request at a time. We have not measured its speed cost (see "Switches that stay
opt-in").

Sampled requests use standard speculative sampling (`MLX_SERVE_DSV41_DRAFT_SAMPLE`), so the output keeps the model's
distribution (the block below has the rule).

<details>
<summary>Sampled drafts and the draft vocabulary</summary>

For a sampled request (`MLX_SERVE_DSV41_DRAFT_SAMPLE`, on), the drafter draws each token from its own distribution q.
The verify keeps it with probability min(1, p / q), where p is the model's distribution, and resamples a rejected token
from max(0, p - q). This is standard speculative sampling, so the output keeps the model's distribution.

The draft vocabulary (`MLX_SERVE_DSV41_DRAFT_VOCAB`) is a `draft-vocab.txt` file in the pack directory. It holds
decimal token ids separated by whitespace; our runs used 65,536 ids. Greedy drafts then score only those ids, plus up
to 16,384 of the request's own most frequent prompt ids (`DRAFT_VOCAB_PROMPT`, on). The trunk and every verify keep the
whole vocabulary, so only the drafts change. A sampled request always drafts over the whole vocabulary. The file gave
+1.46% decode on the standard test. Each draft then reads about half of the LM head and Markov-head bytes.

The converter writes it from `scripts/dsv41_draft_vocab.txt`, beside its builder. That list holds the 65,536 ids that
a general corpus uses most, tokenized with this pack's tokenizer: the CPython standard library, the Zig standard
library, the MLX C++ sources and docs, the macOS SDK C headers and one prose set. The ids for end-of-sequence,
`<think>`, `</think>` and the DSML tool marker are always in it. The source files that the benchmark prompts are built
from were left out of the corpus. On held-out code and prose the list covers 98.4% to 99.6% of tokens.
`scripts/dsv41_draft_vocab.py` rebuilds a list from another corpus. Delete the pack's `draft-vocab.txt`, or
set `MLX_SERVE_DSV41_DRAFT_VOCAB=0`, to draft over the whole vocabulary.

</details>

## Defaults on the M5 Ultra and on other Macs

The plugin's switches are environment variables `MLX_SERVE_DSV41_<NAME>`, registered in `src/env.zig`; a default-on
switch turns off with `=0`. The block below has the other prefixes.

Other Macs get the same defaults except `ATTN_F16`, the lossy f16 attention core, which is on by default without NAX.
Some parts scale or turn off by themselves (listed below).

<details>
<summary>Switch prefixes, and what changes on other Macs</summary>

Four variables use other prefixes: `MLX_SERVE_DSV4_DSPARK` (DSV4, not DSV41), the engine's
`MLX_SERVE_DSPARK_*`, and MLX's `MLX_ENABLE_TF32`. A default-on switch turns off with `=0`. Each speed change in this
release won its A/B on an M5 Ultra with 512 GiB, except `DRAFT_POS_ONCE`, a wash kept because it removes work.

- A Mac that cannot hold the pack refuses the load. The plugin does not stream experts that carry per-expert scales,
  so a smaller Mac cannot serve the NVFP4 pack. It needs the `--experts-mxfp4` pack, whose experts can stream from
  disk (the per-model `stream_experts` setting, see [docs/plugins.md](../../../docs/plugins.md)). Disk reads then limit decode.
- The prompt pass width scales with memory (`PF_PASS_SIZE`).
- Where the Engram tables do not fit in memory, the plugin reads them from disk at each lookup. The switches that need
  every table in memory then turn off, and the kernel family falls back to the `ssd` path (`WEIGHT_PATH`).
- GPUs without NAX (M4 and older) do not run the matrix-unit kernels.

</details>

## Memory and the 1M-token context

The engine admits a load or a request only when the plugin's memory bill fits (its admission check). The KV cache
costs only 3,200 bytes per token (3.36 GB at 1,048,576 tokens); the block below has the bill's four parts.

A 1,032,192-token prompt (the 1M window less a 16,384-token output budget) ran to the end at a 480.5 GB peak, with
the wired limit at 512000 MiB (536.9 GB), and admission billed the whole output budget. At macOS's default limit it is
untested: the memory share is smaller there, so the load picks narrower passes (the block below).

The release's p01 peak is 452.0 GB: 13.4 GB above the first NVFP4 build on the same NVFP4 pack (438.6 GB), and
31.3 GB above it on the MXFP4-experts pack (420.7 GB). The NVFP4 pack adds 17.9 GB, because its 16-value groups carry
more scale bytes. The last reviewed head peaked at 438.6 GB too. Six changes moved the peak by more than
0.5 GB: `PF_KEEP_WIDE` +7.3 GB, the Engram partials cache (`ENGRAM_KV_CACHE`) +4.0, `PF_PASS_SIZE` +3.9,
`MOE_PAIR_FP4` -1.4, `PF_COMP_BF16` -0.7 and `FLASH_ATTN` -0.7. Changes under 0.5 GB each add about +0.7 GB
(`PF_ENGRAM_PASS` +0.4, `DRAFT_VOCAB` +0.4), and drift between A/B bases the last +0.3 GB. The
three that add memory trade it for speed (`PF_PASS_SIZE` +7.96% prefill, `PF_KEEP_WIDE` +2.19% prefill, the Engram
slot kernel and partials cache, one A/B, +0.28% decode). We kept them because
the 1M prompt still fits.

<details>
<summary>The four parts of the memory bill, the 1M fit and its admission</summary>


In the sweep the release peaks 23.7 GB above the ds41nv sweep build at 1K (442.6 against 418.9 GB) and 51.2 GB at
1M (480.5 against 429.3 GB). 20.2 GB is the NVFP4 weights bill. The rest grows with the prompt: the before build's
admission bill grows from 4,683 MiB at 8K to 13,589 MiB at 1M on 8,192-row chunks, while `PF_PASS_SIZE` widens the
release's passes to 44,544 rows (36,087 MiB billed at 1M, with the 18.6 GB `PF_KEEP_WIDE` reserve).

At 1M the whole machine's non-file memory peaked at 489.7 GB (502.0 GB at most over the sweep), against the 536.9 GB
wired limit, with 0 pageouts. At macOS's default limit the 1M point is untested: the memory share is smaller, so
`PF_PASS_SIZE` picks narrower passes than the 44,544 rows it picked at 512000 MiB. At 1M, admission had a 39.0 GB
margin (36,087 MiB needed, 73,320 MiB available) at 512000 MiB. The weights bill is 404.33 GB for the NVFP4 pack against 384.16 GB for the MXFP4-experts
pack (+20.2 GB).

| Part | Where the code computes it | Size for this model |
|---|---|---|
| Resident load | `residentBytes` (`plugin.zig`) calls `residentDiskBytes` (`deepseek_v41.zig`) | The pack's shards less the tensors it drops, plus the Engram tables it holds, the Engram LRU (256 MiB), the Engram partials cache (`ENGRAM_KV_CACHE`, 4 GB) and the draft-only head |
| Compressed KV per token | `kvPerToken` (`bill.zig`) | 3,200 bytes per token |
| Window rings | the rings part of `prefillBytes`'s `state` term (`deepseek_v41.zig`) | 40 layers x (128 + 16) rows x 512 x 2 bytes, about 5.9 MB |
| Prompt pass | `prefillBytes` (`deepseek_v41.zig`), plus the `PF_KEEP_WIDE` reserve | Grows with the pass width. `PF_PASS_SIZE` picks the widest pass that fits the memory share; a set `--prefill-chunk` narrows it |

Layers 2, 8, 14 and 20 are the KV sources. Every layer keeps a 128-row window and reads the compressed rows of the
source layer at or before it. Each source stores a 512-wide latent and a 128-wide index key per token in bf16,
compressed 2:1 in the first three and 1:1 in layer 20.

On the release build, the 1,032,192-token prompt passed the admission check with 36,087 MiB needed and about
73,320 MiB available, with an 18.6 GB `PF_KEEP_WIDE` reserve at 44,544-row passes (the widest pass that fits at 512000 MiB). The Engram partials cache (`ENGRAM_KV_CACHE`) allocates
3,787 MiB (3.97 GB) of its 4.0 GB cap on this pack. The reserve replaces allocations that the pass made anyway, so the
peak rises by 7.3 GB, not by the reserve's size.

`PF_KEEP_WIDE`'s reserve needs `patches/mlx-metal-wide-reserve.patch`, which adds a reserve of large Metal buffers to
MLX's allocator. A prompt pass then does not map fresh pages in every request. Without the patched libmlx, the plugin
declines the switch by name and logs why.

</details>

## Accuracy: what "lossless" means here

54 plugin switches fall in three classes, and a fourth class, TF32, covers one engine variable:
- EXACT: the same output values as the `=0` path, bit for bit. A test shows it, or (2 switches) it is classed without
  a measurement, by the commit's reasoning.
- REORDER: not bit-identical; the table below splits the 17 by their evidence.
- DRAFT-ONLY: changes only the drafts, or how many rows a verify checks (1 switch classed without a measurement).
- TF32 (one engine variable): MLX's TF32 matmuls; on off-default routes they also round the f32 chain's P·V (the
  block below). Without NAX, MLX does not use TF32.

48 of the 54 have a standing test or a kernel contract case. 6 do not: `WOA_JOIN`, `DRAFT_POS_ONCE` (EXACT); `CED`,
`PF_PASS_SIZE`, `HC_EXPAND_NORM` (REORDER); and `SPEC_SYNC` (DRAFT-ONLY). The table below says why for each.

| Class | Plugin switches | Engine variables |
|---|---|---|
| EXACT | 31 | 0 |
| REORDER | 17 | 1: `MLX_SERVE_DSV4_DSPARK` (the verify's multi-row kernels, by reasoning) |
| DRAFT-ONLY | 6, including `DRAFT_VOCAB` | 2: `MLX_SERVE_DSPARK_POLICY2`, `MLX_SERVE_DSPARK_POLICY2_MODELED` |
| TF32 | 0 | 1: `MLX_ENABLE_TF32` |

The two exceptions, both on in `main`, are `HEAD8` (the 8-bit LM head in "What the pack holds") and
`ATTN_F16`, an attention core with f16 scores and probabilities. One function, `attnCore`, picks the core for every
prompt-attention call (the block below).

<details>
<summary>The classes, the attention core, the tests and their results</summary>

| REORDER switches | How many |
|---|---|
| Measured no farther from float64 than the `=0` path (`PF_COMP_BF16`, `FLASH_ATTN`, `ENGRAM_KV_SLOTS`, `ATTN_DEC`, `WOA_PLAIN`, `PF_IDX_PASS`) | 6 |
| Held to a stated bound that can allow more error than `=0`: `MOE_FP4_NAX`, `FP4_READS`, `FUSE_MXFP8_QMM` (rel-rms against f32 up to 1.25 times the `=0` path's, plus 1e-4); `PF_HC_V2` (mixes and split 2.0 to 5.9 times farther from float64, within 1e-5 of it); `ENGRAM_GPU_IDS` (4 of 409,600 words farther, worst element no farther); `MOE_PAIR_FP4` (each differing word within one bf16 step of its row's peak, under 1% differing, sample l2 within 1.05 times); `PF_IDX_NAX` (each score within 2·γ(34) of `idx_tile`'s, the same top-half picks except near-ties) | 7 |
| Resting on the commit's reasoning (`CED`, the chunked prompt path; `PF_PASS_SIZE`) | 2 |
| Resting on its A/B's same text (`HC_EXPAND_NORM`: its test allows 1% relative error, far looser than bf16 rounding, so we do not count it as a bound; `PF_PASS_CEIL`, a different partition into passes: 28 of 28 sweep pairs from 128K to 1M gave the same text) | 2 |

A DRAFT-ONLY switch keeps the verify's rule: a greedy verify keeps only tokens that the model agrees with, and a
sampled verify keeps the model's distribution. Greedy text can still change by rounding, because the verify kernels'
sum order depends on the verify width. At the defaults on M5, TF32 rounds no input of the model's own matmuls, only
the drafter's P·V, so it passes for the same reason as DRAFT-ONLY.

The tests measure error, not formats. The flash kernel, the gather (the plugin's three-launch attention core on the
matrix units, `FLASH_ATTN`'s `=0` path) and the f32 chain (MLX's matmuls and softmax in f32) under TF32 all round the
attention probabilities to a 10-bit mantissa. `FLASH_ATTN` passes, because its output is nearer to float64 than the
gather's. `ATTN_F16` fails, because its f16 scores and probabilities are 10% to 37% farther.

`81d0b394` fixed multi-span calls; one-span calls still reached the f16 core on M5 until `4452a4bd` gave the choice
one owner, `attnCore`, with tests on the served routes of a full-model fixture.

One function, `attnCore`, picks the core for every prompt-attention call. It decides from the call's shape, dtypes and
device and the load's switches only, never from the caller (a debug trace also turns off flash). On NAX it takes the first core that fits:

1. the flash kernel, where `FLASH_ATTN` is on and the head shape fits, at any pick count up to 896 (window plus picks
   within 1,024);
2. the gather, where the pick count (the compressed rows that the indexer picks) is a multiple of 64;
3. the f32 chain (MLX's matmuls and softmax in f32).

Without NAX the plugin has no flash kernel and no gather, because both need the matrix units. A call that fits the
f16 core's 1,024-column softmax takes the f16 core, or the f32 chain under `ATTN_F16=0`; the f32 chain takes the rest.
The f16 core was mlx-serve's default there before this release. We kept it, because we have not measured the speed or the error of the f32 chain (`=0`) on those Macs. On NAX the f16 core serves only under an
explicit `ATTN_F16=1`. The load fixes the switch's value, and the prompt cache's layout hash includes it.

TF32 rounds an f32 matmul's operands to a 10-bit mantissa. On M5 at the defaults, every prompt-attention call takes
the flash kernel or the gather, and the trunk's f32 matmuls read bf16 values held in f32 or run on MLX's gemv kernel.
So TF32 rounds no operand that the model verifies. The f32 chain serves only off-default routes, and there TF32
rounds its P·V; its float64 error measured equal to the f32 chain's own error on the default bf16 route at 64
heads, at those shapes. γ(n) = n·u / (1 - n·u), with u = 2⁻²⁴, bounds the rounding of an n-term f32 sum. At the defaults TF32's
only rounding is the drafter's P·V, so its class passes for the same reason as DRAFT-ONLY.

The class tests are in the plugin unless named. Most also run a planted fault as a control, which must fail.
`zig build test -Dslow-tests -Dtest-filter="dsv41" -Dtest-filter="deepseek_v41" -Dtest-filter="dspark sampled verdict"`
runs them all; `prompt/wide_test.zig` holds most prompt switches. `decode/attn.zig`, `decode/woa_plain.zig` and
`decode/engram_ids.zig` hold `ATTN_DEC`, `WOA_PLAIN` and `ENGRAM_GPU_IDS` against float64. `tf32_test.zig` runs the
served f32 matmuls against float64, over operands saved from a served run (`DSV41_TF32_FIXTURE`). The served-route
tests of `attnCore` count each core's calls; on NAX they count 0 f16 calls.

- A tensor split (32 heads per rank) takes the f32 chain. The f32 chain also serves `FLASH_ATTN=0` at a pick count that
  is not a multiple of 64, and f16 or f32 activations. On those routes attention's output depends on the prompt pass
  width, because MLX picks its matmul by row count. This is a reorder. Flash and the gather do not depend on it.

"bf16-out l2" and "f32-out l2" below are the relative l2 error against float64 of the bf16 or f32 output.

- `ATTN_F16` (LOSSY): at the release's prompt spans on M5, bf16-out l2 1.80e-3 to 2.31e-3, against 1.63e-3 to
  1.68e-3 for the f32 chain.
- The flash kernel at those spans: bf16-out l2 1.660e-3 to 1.670e-3 against 1.667e-3 to 1.677e-3 for the f32 chain,
  and f32-out l2 6.2e-5 to 6.6e-5 against 1.52e-4 to 1.67e-4.
- `MOE_TAIL` (EXACT): one launch gives the two launches' words, with 0 differing words at 17 to 8,192 rows.
- `FLASH_ATTN` (REORDER): its f32-out l2 is 6.4e-5, against 1.8e-4 for the gather.
- `MOE_PAIR_FP4` (REORDER, bounded): one switch for both fp4 formats. NVFP4 brings ModelOpt's per-expert scales into
  the epilogue; MXFP4 (or the MXFP4 view) runs the same kernel with that step compiled out. It replaces
  `MOE_PAIR_SCALED` (NVFP4 only), which no release on main carried. The test requires each differing word within one
  bf16 step of its row's largest value, under 1% of words differing, and a sample l2 within 1.05 times the split
  path's. At 384 experts:
  - NVFP4: 0.223% to 0.237% of the activation words differ from the split path, and the whole-sample float64 l2 is
    1.0000 times the split path's. The words more than one step apart in bf16 code are words that the split path
    gives as exactly 0.0, where the paired kernel keeps a tiny value, about 1e-35.
  - MXFP4: 0.0014% to 0.0017% of the words differ, none more than one bf16 code step apart, and the whole-sample
    float64 l2 is 1.0000 to 1.0008 times the split path's. The planted control (each pair's routing weight on the
    next pair) is 233 to 325 times farther.
- `PF_HC_V2` (REORDER, bounded): an f32 reorder of the five-pass hyper-connections. Its f32 mixes and split are 2 to 6
  times farther from float64 than the five passes', at the 1e-7 level. That is below the bf16 rounding that follows,
  and the tail's output words are the five passes' words. The test holds the mixes and split within 1e-5 of float64,
  and every other output within 1.01 times the five passes' l2.
- `ATTN_DEC` (REORDER): over 72 cells, nearer to float64 than `dec_attn` (`=0`) in total l2 and worst element. Its
  window-only keys run 2 splits of 64: l2 / max 1.599e-7 / 4.056e-7 against 1.632e-7 / 4.439e-7 with the float64 merge,
  and 1.688e-7 / 4.141e-7 against 1.769e-7 / 4.634e-7 with the f32 merge.
- `WOA_PLAIN` (REORDER): nearer to float64 than its `=0` path in total l2 and worst element.
- `ENGRAM_GPU_IDS` (REORDER, bounded): 30 of 409,600 bf16 words differ from MLX's chain: 25 nearer, 4 farther and
  1 at the same distance. The worst element is equal, and the total squared error is larger by a factor of
  1 + 4.5e-9. The test requires a worst element no farther and at most 1e-4 of the words differing.
- `FP8_FOLD` and `FP4_READS` read three rare mxfp8 code kinds (e8m0 scale 0xFF, e8m0 scales 247 to 254, and the e4m3
  NaN) differently from MLX. At load, the plugin scans each layer:
  it declines `FP8_FOLD` by name where any mxfp8 linear holds one, and `FP4_READS` where a shared down does. The shipped NVFP4 pack holds none of them.
- TF32: the router, the compressor projection and the DSpark q·k read 0 operand elements that TF32 rounds. The hc
  collapse has operands that TF32 would round, but MLX serves it on its gemv kernel, which does not use TF32, so its
  error is the same with TF32 on and off. The only TF32 loss is the drafter's P·V, at a relative l2 of 2.8e-4 to
  2.9e-4, and it changes only the drafts.

</details>

<details>
<summary>Every default-on switch and its class</summary>

Names omit `MLX_SERVE_DSV41_`. "Evidence" names the evidence kind: a standing test (in the plugin unless named), the kernel contract,
or the A/B's same output text ("A/B text"), or the commit's reasoning ("commit").

| Switch | What it does | Class | Evidence |
|---|---|---|---|
| PROMPT_ASYNC_EVAL | Commits the last wide prompt layer without a wait | EXACT | test (word for word) |
| PF_CED_HC | CED compressor hc pre-norm once over the pass | EXACT | test (word for word) |
| PF_ROPE_PAIRS | Query-tail rope computes each angle once per pair | EXACT | test (word for word) |
| PF_RMS_ROWS | Query-latent rms reads its rows in place | EXACT | test (word for word) |
| CED_LAZY | Runs a CED replay once, for its first reader | EXACT | test (word for word) |
| IN_PROJ_MIXR | Verify in_proj hc mixes as one threadgroup per (mix, row) | EXACT | test (word for word; M3 or later) |
| HC_EXPAND_NORM | FFN-side hc expand and norm in one launch | REORDER | A/B text (its test allows 1% relative error, not counted as a bound) |
| PF_MOE_EPI | Counting-sort route pack in place of two argsorts | EXACT | test |
| PF_GU_SEG | 128 x 128 paired gate/up where the planner picks it | EXACT | test (word for word, to 8,192 rows, NVFP4 with per-expert scales; NAX) |
| PF_KV_ROWS | Window kv rows on one simdgroup per row | EXACT | test (word for word) |
| PF_COMP_PASS | Compressor pooling once over the pass, each span's index-key linear on its own groups | EXACT | test (word for word) |
| PF_CED_COMP | The CED compressor at the split layer: each span's projection as before, then its norm, index keys and rope over the pass in one compressPass | EXACT | test (word for word) |
| PF_DOWN_TOKORD | Staged down writes rows in token order | EXACT | test |
| PF_DOWN_BK32 | Staged down on the 64 x 64 tile, 32 deep a step, at 96 routed rows an expert and more | EXACT | test (word for word) |
| PF_FP4_GEMM | Staged matrix-unit gather for routed gate/up and down | EXACT | test |
| WOA_JOIN | Joins the spans' wo_a rows in one copy | EXACT | commit |
| PF_ROUTE_PACK | Router top-6 on a simdgroup per row; one-threadgroup scan | EXACT | test |
| PF_ROUTER_BF16 | Router logits on a bf16 MMA, only where MLX runs one sum (1,707 rows and more) | EXACT | test |
| PF_KEEP_WIDE | Prompt buffers from the MLX patch's reserve | EXACT (memory) | test |
| PF_ENGRAM_PASS | Engram wkv once per 16 spans | EXACT | test |
| MOE_TAIL | MoE combine and hc expand in one launch | EXACT | test |
| FLASH_TRIM | FLASH_ATTN with a raw-score block max and 8-byte stores | EXACT (to FLASH_ATTN) | test |
| FFN_IN_SINK | ffn_in mix tail in its own launch | EXACT | test (word for word) |
| SPEC_ENGRAM_AHEAD | Queues the next verify's Engram kv behind the draft | EXACT | test |
| FFN_IN_R5_COLS1 | 5-row ffn_in launch 1 with one column per simdgroup | EXACT | test (word for word) |
| DRAFT_POS_ONCE | Makes draft ring slots and positions once | EXACT | commit |
| FUSE_WQB_RING | The wq_b launch writes the window ring | EXACT | contract |
| ENGRAM_KV_CACHE | GPU cache of Engram slot partials, at most 4 GB | EXACT (to ENGRAM_KV_SLOTS) | test |
| FP4_DEC_WORD | Decodes e2m1 codes a word at a time | EXACT | test |
| FP4_UP_SPLIT | Shared gate and up rows on separate simdgroups | EXACT | test |
| FP8_FOLD | Folds 2^8 into the e8m0 scale in mxfp8 reads | EXACT (declines on rare codes) | test (word for word on finite codes), load scan |
| MOE_DOWN_ORDER | Verify down launch in column-block order | EXACT | test (bit for bit, R = 2 to 6) |
| FUSE_DOWN_SCALE | NVFP4 down scale inside the gate/up launch | EXACT | test (word for word) |
| CED | The chunked prompt path: the decoder layers run again over a chunk's last window rows only | REORDER (row count) | commit (its fixture test is bit for bit only at 16-row chunks) |
| MOE_FP4_NAX | Fused fp4 MoE decode kernels on the matrix units | REORDER (bounded) | test (within 1.25x of the `=0` path's rel-rms against f32, + 1e-4) |
| MOE_PAIR_FP4 | fp4 prompt gate/up (NVFP4 and MXFP4), the scales (NVFP4) and activation in one launch | REORDER (bounded) | test |
| PF_HC_V2 | Wide-layer hc in three passes, not five | REORDER (bounded) | test |
| PF_PASS_SIZE | Near-equal memory-sized prompt passes | REORDER (partition) | commit |
| PF_PASS_CEIL | A memory-sized prompt pass's working set capped at the bytes of the knee measured on the M5 Ultra 512 GiB (the bill of 39,680 rows; a narrower memory-sized pass is kept) | REORDER (partition) | test (the plan) and A/B text (the view at 1M +3.27%, 512K +0.84%; NVFP4 already plans under it) |
| PF_IDX_NAX | Indexer scores on the matrix units | REORDER (bounded) | test (within 2·γ(34) of `idx_tile`, the same top-half picks except near-ties; NAX) |
| PF_COMP_BF16 | Compressor projection on a bf16 MMA, K blocks summed apart | REORDER | test |
| PF_IDX_PASS | Indexer queries and head weights once over the pass, the head weights on a shape-cut kernel (16 partial sums) | REORDER | test (head weights no farther from float64 than MLX's GEMM per span; +0.8 GB peak, billed) |
| FLASH_ATTN | Every prompt-attention call on one flash kernel where it fits | REORDER | test |
| ENGRAM_GPU_IDS | Engram ids hashed on the GPU; fused wkv | REORDER (bounded) | test |
| ENGRAM_KV_SLOTS | Slot-blocked Engram wkv | REORDER | test |
| FP4_READS | Bit-placement fp4 reads; each 16-group summed apart | REORDER (bounded; declines on rare codes) | test (within 1.25x of the `=0` path's rel-rms against f32, + 1e-4) |
| ATTN_DEC | Decode attention on a shape-cut tile kernel | REORDER | test |
| WOA_PLAIN | Splits merged once; wo_a as a plain grouped GEMV | REORDER | test |
| FUSE_MXFP8_QMM | Prompt mxfp8 linears on MLX's quantized matmul | REORDER (bounded) | test (within 1.25x of the `=0` path's rel-rms against f32, + 1e-4) |
| DRAFT_FUSED | DSpark stages on the fused decode kernels | DRAFT-ONLY | test (within 1% relative rms of the op chain) |
| DRAFT_SAMPLE | Sampled drafts from q, kept at min(1, p / q) | DRAFT-ONLY | test (engine: `spec/dspark.zig`, "dspark sampled verdict") |
| DRAFT_QKV | Fuses a stage's wq_a and wkv at load | DRAFT-ONLY | test (within 1e-3 relative rms, no farther from the op chain) |
| SPEC_SYNC | Greedy accept on the GPU; next draft queued behind the verify | DRAFT-ONLY | A/B text (its test needs a pack) |
| DRAFT_VOCAB | Greedy drafts score only the pack's `draft-vocab.txt` ids (default: the pack's file, which the converter writes) | DRAFT-ONLY | test (and `deepseek_v4/served_test.zig`) |
| DRAFT_VOCAB_PROMPT | A request's prompt ids join the draft vocabulary | DRAFT-ONLY | test (`deepseek_v4/served_test.zig`) |
| HEAD8 | LM head requantized to 8-bit g64 | Requantized (KLD within one standard error) | KLD measurement |
| ATTN_F16 | f16 attention core; on by default only without NAX | LOSSY (measured, not gated) | printed float64 measurement, served-route tests |

Engine variables on the DeepSeek path: `MLX_SERVE_DSV4_DSPARK` (REORDER, by reasoning: the verify's multi-row kernels),
`MLX_SERVE_DSPARK_POLICY2` and `MLX_SERVE_DSPARK_POLICY2_MODELED` (DRAFT-ONLY: they pick verify widths) and
`MLX_ENABLE_TF32` (TF32).

</details>

## Switches that stay opt-in

Four switches stay off by default (`EXACT`, `FLASH_PSUM`, `MXFP4_VIEW`, `PF_INT8`), and three value switches set
memory and kernel choices. `MLX_SERVE_DSV41_EXACT=1` now also serves NAX; `MLX_SERVE_DSV41_EXACT_VERIFY`, which only
earlier heads of PR 1256 had, is gone.

<details>
<summary>The opt-in and value switches</summary>

Each of these is off by default. Names omit `MLX_SERVE_DSV41_` except where shown.

- `EXACT=1|solo` (EXACT) makes greedy DSpark text equal plain decode's bytes, one request at a time. Any value other
  than unset, empty or `0` turns it on. On NAX, it runs each verify row's head, Engram wkv and wo_a as the plain step
  does. On M3-generation or later Apple GPUs without NAX the whole verify row does. `solo` also makes every request
  decode alone, and it is served only without NAX. We have not measured its speed cost.
- `FLASH_PSUM=1` (REORDER) is a flash-attention variant that is nearer to float64. It measured no faster.
- `MXFP4_VIEW=1` (REORDER) loads an NVFP4 bank whose every 32-value span is one power of two as MXFP4. On the
  release build the greedy A/B decides: the view is -1.70% prefill and -0.33% decode there, and its use is the
  16.8 GB of peak it saves (452.3 to 435.5 GB). With the card's settings, for the record: -1.62% prefill, +3.85%
  decode.
- `PF_INT8=3|4` (LOSSY; single Mac, M5 Ultra 512 GB) runs a wide prompt pass's trunk projections in the 20 CED
  encoder layers on w8a8's rotated int8 GEMM (one f32 scale per row on each side). Level 3 takes `wq_b` and `wo_b`.
  Level 4 also takes the shared expert and the fused `wq_a`+`wkv`. The int8 weights are made at load and billed
  (1.68 GB at level 3, 2.57 GB at level 4). The decoder layers stay exact. A two-Mac pair, a tensor split, streamed
  weights and Macs without NAX run the exact path, and the load says so. On the
  16K standard test, level 4 is +11.48% prefill (5,591.8 to 6,233.9 tok/s) with the same decode. Its 16K continuation
  KLD against the exact path is 0.00834, and the `FLASH_ATTN=0` path's is 0.00829 (a difference of z +0.12).

Three value switches set memory and kernel choices. `ENGRAM_LRU_MB` (256) sets memory only. `ENGRAM_RESIDENT` (`auto`)
and `WEIGHT_PATH` (`auto`) can turn off the resident fusions. The `ssd` path turns off `FUSE_DOWN_SCALE` and `FUSE_MXFP8_QMM`. Tables not
held in memory turn off `ENGRAM_GPU_IDS` and the Engram partials cache (`ENGRAM_KV_CACHE`).

</details>

## Results

The first NVFP4 build is `f7e4da9e`, an early build of this PR's branch (2026-11-20), before the fused fp4 decode was
on by default. `main` cannot serve this pack: it has neither the converter nor a `deepseek_v41` runner that reads the experts'
per-expert scales.
Standard test, p01, greedy, DSpark on, NVFP4 pack:

| Build | Prefill tok/s | Decode tok/s |
|---|---|---|
| `f7e4da9e`, the first NVFP4 build | 3,994.1 | 71.77 |
| `58279979`, the last reviewed head | 4,009.6 | 82.93 |
| `a99fc4f6`, the release | 5,551.3 | 98.80 |

Four later prompt levers (`PF_COMP_PASS`, `PF_IDX_PASS`, `PF_DOWN_BK32`, `PF_CED_COMP`) are on by default and not in
these cells; their gains are measured per A/B (+0.27%, +0.26%, +0.08% and +0.02% prefill; `PF_CED_COMP`'s A/B gave
the same text in both arms and is kept for the about 50 launches a span it removes). From `58279979` to the release (the 47 changes, the upstream merge and the audit fixes): +38.4% prefill and +19.1% decode (`58279979` on
Mac 1, the release on Mac 2, which runs about 0.8% slower per round). On `f7e4da9e` the MXFP4-experts pack ran
4,108.3 / 73.83 tok/s; NVFP4 was 2.8% slower.

The ds41nv sweep build (see "Terms") ran the same sweep on the MXFP4-experts pack on 2026-11-24. Greedy prefill /
decode tok/s:

| Context | The ds41nv sweep build, MXFP4 | Release, NVFP4 | Prefill, release / before | Decode, release / before |
|---|---|---|---|---|
| 16K | 4,248.0 / 102.65 | 5,524.1 / 132.80 | x1.30 (x1.27 against seed 4321 alone) | x1.29 (x1.26) |
| 128K | 3,915.8 / 103.05 | 5,508.9 / 128.45 | x1.41 | x1.25 |
| 512K | 2,728.6 / 102.30 | 4,707.5 / 119.00 | x1.73 | x1.16 |
| 1M | 1,889.5 / 99.75 | 3,912.9 / 110.70 | x2.07 | x1.11 |

The sweep's code-context prompts reach 4.4 to 4.9 tokens per round, against 2.99 on the standard test, so do not
compare sweep decode with standard-test decode.

The standard test is three coding prompts of about 17,600 tokens each, with 1,024 completion tokens and a fresh server
for each run. Each cell ran 6 runs (p10: 2) on the release build `a99fc4f6`, started with `--ctx-size 32768
--warmup-full`. The card's sampling settings are temperature 1.0 and top_p 0.95.

| Cell | Prefill tok/s | Decode tok/s | TTFT s | Peak GB | Tokens per round |
|---|---|---|---|---|---|
| Greedy, DSpark on | 5,526.4 | 97.98 | 3.20 | 452.3 | 2.99 |
| The model card's sampling settings, DSpark on | 5,535.4 | 88.05 | 3.19 | 452.3 | 2.70 |
| Greedy, DSpark off | 5,546.6 | 61.33 | 3.19 | 452.3 | - |
| Prompt p10, greedy, DSpark on | 5,524.5 | 98.50 | 3.07 | 451.8 | 2.94 |

<details>
<summary>Notes on the cells</summary>

The runs differ. The first NVFP4 build's A/B used a 440320 MiB wired limit and plain pre-warm requests. The release
cells used 512000 MiB, `--ctx-size 32768` and `--warmup-full`. p01 decodes fastest of the three prompts; the
release's three-prompt mean is 97.98 tok/s.

The before build's 16K greedy seed-1234 run was low (4,149.4 / 99.8 against 4,346.7 / 105.5 for seed 4321), as was its
1K card seed-1234 run; the means include them.

The release's attention fix sends the CED replay of a long prompt to the flash kernel, so greedy texts differ from the
control's (`04b8f7d0`). On p10 the release decodes 7.5% slower than the control (98.50 against 106.45 tok/s): acceptance fell from
3.51 to 2.94 tokens per round on the new text, and round time did not rise. p10 is the only cell whose mean decode fell
more than 1%. Per prompt, greedy p21 fell 3.1% and p14 with the card's settings 3.9%, while greedy p14 rose 4.1% and
card p01 2.9%; every one of these texts changed. The PR's reply comments hold every run and each switch's A/B.

A 17,600-token prompt runs as one prompt pass at the cells' context and at the 1M default (the 1M load cuts passes
of about 44,500 rows), but we have not measured the cells at the 1M default.

</details>

## Gaps

| Gap | What the release needs |
|---|---|
| The plugin does not stream experts with per-expert scales, so a Mac that cannot hold the NVFP4 pack cannot serve it | The `--experts-mxfp4` pack for those Macs, or streamed per-expert scales |
| Without NAX, the lossy f16 attention core (`ATTN_F16`) is on by default, and we have not measured it on those Macs | A float64 test on those Macs, or `ATTN_F16` off there |
| On the f32 attention chain's routes, attention's output depends on the prompt pass width | A test that holds each pass width no farther from float64 |
| 6 classed switches have no standing test against their `=0` path (listed in "Accuracy") | A standing word or float64 test for each |
| TF32's float64 test needs a fixture that no tool in this PR writes; ours came from the `DSV41_TF32_AUDIT` dump on branch `rel-harden-tf32` (`6965929b`), not in this PR | A fixture writer, or the fixture in the repo |
| `HEAD8` is a quantization change. Its KLD is within one standard error of the bf16 head's on our set, but it is not bit-identical | A ruling to keep it, or `=0` by default |
| We tuned and measured only on M5 Ultra Macs with 512 GiB | Runs on smaller Macs and older GPUs |
| The release has not loaded at the default wired limit | One load at the default limit |
| We have no PPL or KLD of the whole stack; the KLD above is of the LM head alone | A whole-stack KLD against the same reference |
| We have not measured `EXACT=1`'s speed cost | An A/B with `EXACT=1` |
| No speed run on the PR head: its later commits are checked by tests and a text check only | The standard test on the PR head |

## Terms

| Term | Meaning |
|---|---|
| NVFP4 | NVIDIA's 4-bit float format: e2m1 values, one e4m3 scale per 16, plus a per-tensor global scale |
| MXFP4, mxfp8 | The OCP microscaling formats: one e8m0 scale per 32 values |
| DSpark | DeepSeek-V4.1's own drafter stages, used for speculative decoding |
| Verify | One step of the model over the current token and the drafts |
| Engram | DeepSeek-V4.1's hashed n-gram embedding tables |
| hc | DeepSeek-V4.1's hyper-connections, the multi-stream residual |
| CED | The plugin's chunked prompt path: the decoder layers run again over a chunk's last window rows only (the CED replay) |
| NAX | The M5 GPU's matrix units; M4 and older Macs have none |
| Span | The 256-row unit that a wide prompt layer's attention takes |
| The gather | The plugin's three-launch attention core on the matrix units, `FLASH_ATTN`'s `=0` path |
| The f32 chain | MLX's matmuls and softmax in f32, the attention fallback |
| Admission | The engine's check that the memory bill fits |
| Control | Build `04b8f7d0`, the stack before the upstream merge and the release fixes |
| Pick count | The number of compressed rows that the indexer picks for a span's attention |
| KLD | Kullback-Leibler divergence of the served distribution from a reference |
| `=0` path | The path a default-on switch serves when it is set to 0: the path before the change |
| Standard test | Three coding prompts of about 17,600 tokens, 1,024 completion tokens (greedy, or the card's sampling settings), a fresh server per run, 6 runs |
| p10 | A fourth library prompt, not part of the standard test; it drafted best on the control |
| The first NVFP4 build | `f7e4da9e`, an early build of this PR's branch (2026-11-20), before the fused fp4 decode was on by default |
| The ds41nv sweep build | The stack at the start of this work: `main` `9e3311c2` plus the Engram mxfp4 path and its residency, MXFP4-experts pack; binary sha256 `534d15b3`, no commit recorded |
| Word | The bits of one output element (one bf16 or f32 value) |
| Last reviewed head | `58279979`, the PR 1256 head that reviewers saw before this release |
