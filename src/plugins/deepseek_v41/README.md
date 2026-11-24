# DeepSeek-V4.1-Flash-1017 NVFP4 in mlx-serve

Status: release documentation for the `deepseek_v41` runner in mlx-serve core, written for PR 1256 (head `53ed5e05`).
"We" are the PR 1256 authors. We tuned and measured on two M5 Ultra Macs with 512 GB of memory. Agent and code
notes for this folder are in [AGENTS.md](AGENTS.md).

<!-- Sources for this draft are in HTML comments beside each section. Paths are relative to the repo root. -->

## Summary

This plugin serves NVIDIA's `DeepSeek-V4.1-Flash-1017-NVFP4` checkpoint on Apple GPUs. A converter script rewrites
only the checkpoint's file headers, so the weights keep NVIDIA's bytes. The routed experts stay in NVFP4. The
default configuration is lossless against the pack as shipped, with one stated exception: the plugin requantizes
the LM head to 8 bits, and that head measured the same KLD as the bf16 head. Every other default-on switch except
`ATTN_F16` (on only without neural accelerators, see "Gaps") is in one of four classes: EXACT, REORDER, DRAFT-ONLY or TF32. We tuned the defaults on an M5 Ultra with 512 GB. Other Macs get
the same defaults, and a Mac that cannot hold the pack refuses the load. On that machine the full 1,048,576-token
context fits. DeepSeek's own drafter (DSpark) speeds up decode. HumanEval, the long-context sweep and the final
numbers on the release head are not done yet.

<!-- src: src/plugins/deepseek_v41/AGENTS.md (head paragraph); scripts/dsv41_nvfp4_pack.py docstring;
     src/plugins/deepseek_v41/deepseek_v41.zig:2584 (head requantize log), :883 head8Wanted;
     /tmp/ds41sweep/lanes.txt 2026-11-23T20:32:26Z (KLD); /tmp/ds41sweep/release/audit.md section 1 and 3 -->

## What the runner serves

**The pack.** `scripts/dsv41_nvfp4_pack.py SRC_DIR DST_DIR [--check] [--experts-mxfp4]` converts NVIDIA's checkpoint
into a pack that this plugin reads. It makes each shard an APFS clone and rewrites the header at the same length, so
no tensor data is copied. Run it on the machine that holds the checkpoint. `--check` reads the result again and
compares each converted tensor with the source. The converter does these things:

- The fp8 linears become MLX mxfp8 at group 32. One side shard holds their row scales.
- The DSpark stages' routed experts become MLX mxfp4 at group 32.
- The Engram tables (DeepSeek-V4.1's hashed n-gram embeddings) stay mxfp4, as stored.
- The routed experts stay NVFP4 (MLX nvfp4 at group 16). ModelOpt's per-expert global scales move to a second side
  shard, `model-expert-scales.safetensors`. The gate scale applies before the SiLU, the up scale before the SwiGLU
  clamp, and the down scale on the routing weight.

<!-- src: scripts/dsv41_nvfp4_pack.py lines 1-45 -->

**Why NVFP4.** NVFP4 is the format NVIDIA ships. Its 16-value groups and e4m3 scales hold spiky weights better than
MXFP4, so in general it is the more accurate format. `--experts-mxfp4` writes the experts as MXFP4 instead. That form
is value-identical to NVIDIA's for this checkpoint only, because all of this checkpoint's scales are powers of two.
The converter checked 16,986,931,200 groups. The MXFP4 form was faster and 18 GB smaller in an earlier comparison.
The plugin reads either form from the pack, with no flag.

<!-- src: scripts/dsv41_nvfp4_pack.py docstring (--experts-mxfp4 paragraph); src/plugins/deepseek_v41/AGENTS.md
     head paragraph (16,986,931,200 groups); /tmp/ds41sweep/pr/1256-body.md "NVFP4 against MXFP4" (4,108.3 vs
     3,994.1 tok/s prefill, 420.7 vs 438.6 GB) -->

**The LM head (`MLX_SERVE_DSV41_HEAD8`, on).** The pack ships the LM head in bf16. At load the plugin requantizes it
to 8-bit affine at group 64. The trunk and the drafter share this head. This is the one quantization change in the
release defaults. The reference is the top-20 log-probabilities that DeepSeek's API returned on 2026-09-10 for
`deepseek-flash` (the ds4 repo's `deepseek-v4.1-flash-20260910-general` set; its README names weights revision
`df42c109`). That reference can differ from the 1017 checkpoint, so the absolute KLD includes any difference between
the two models, and the paired difference measures the head. On 50 prompts and 1,600 positions, the 8-bit head's
mean KLD was 0.1020 and the bf16 head's was 0.1021. The paired difference was
+0.00007 ± 0.00025 (one standard error), and top-1 agreement was 92.6% for both. The bf16 head (`=0`) cost 4.95% of
spec-on decode speed and used 0.63 GB more memory.

<!-- src: src/plugins/deepseek_v41/deepseek_v41.zig:2583-2584 (quantizeLin(m, w, m.head, 8, 64)), :883-890;
     m1:~/glmx/abba/ds41-diag-kld-head16/a.summary.txt, b.summary.txt (build 2b0bba65, 1600 positions, paired
     +0.00007 +/- 0.00025); m1:~/github/ds4 git show HEAD:gguf-tools/quality-testing/
     deepseek-v4.1-flash-20260910-general/README.md (API 2026-09-10, top-20 logprobs, revision df42c109);
     m1:~/glmx/abba/ds41-ab-sp-head16/summary.txt:22-25 (94.33 -> 89.67 tok/s, -4.95%, peak +0.63 GB) -->

**The MLX patch.** `MLX_SERVE_DSV41_PF_KEEP_WIDE` needs `patches/mlx-metal-wide-reserve.patch`. The patch adds a
reserve of large Metal buffers to MLX's allocator, so a prompt pass does not map fresh pages in every request. Without
the patched libmlx, the plugin declines the switch by name and logs why.

<!-- src: src/env.zig:369 (PF_KEEP_WIDE doc); /tmp/ds41sweep/pr/1256-body.md "The MLX patch" -->

## Set up an M5 Ultra with 512 GB

Do these steps in this order.

1. Convert the checkpoint with the converter above.
2. Put `draft-vocab.txt` in the pack directory (see "Speculative decoding"). Without it, the drafts score the whole
   vocabulary and decode is slower.
3. Raise Metal's wired-memory limit to 512000 MB (500 GiB): `sudo sysctl iogpu.wired_limit_mb=512000`. Every run
   on this release's stack used it, the A/Bs and the final cells. The value does not survive a restart: `sysctl`
   sets it for the running system only, so set it again after each restart.
4. Start the server with `--warmup-full`, for example
   `mlx-serve --model <pack> --serve --warmup-full`.

<!-- src: box-*.txt in every ~/glmx/abba/ds41-ab-* and ds41-final-* result directory on m1 and m2
     (iogpu.wired_limit_mb: 512000; the last push's nvpf-dsproof-ab used 440320); neither box has /etc/sysctl.conf
     or a launch daemon that sets it;
     docs/cli.md:115 (--warmup-full); docs/cli.md:31 (--model --serve) -->

**Why the wired limit.** Metal's working-set limit can refuse a load that RAM holds. The engine then names
`iogpu.wired_limit_mb` in the error and logs the `sudo sysctl iogpu.wired_limit_mb=<N>` that admits the load. N is
the load's need (the weights plus warm-up headroom) plus an 8 GiB margin. The last push's build loaded at macOS's
default limit and peaked at 438.6 GB (from that push's PR body). Control `04b8f7d0` peaks at about 452.3 GB on our
standard test, and we have not tried this release at the default limit.

<!-- src: src/engines/mlx/load.zig:149-153; src/engines/mlx/runtime/preflight.zig:326 (wiredLimitAdviceMb);
     /tmp/ds41sweep/pr/1256-body.md "Basic config on a 512 GB M5 Ultra" and "Not measured, and open" -->

**Why `--warmup-full`.** Metal compiles a kernel at its first use. With `--warmup-full`, the server serves itself a
ladder of prompt sizes before it reports ready, so no user request pays a first compile. This takes from seconds to
a minute, on a build's first load of the model only. The final cells and the last A/Bs used it; the earlier A/Bs
relied on unmeasured pre-warm requests instead.

<!-- src: docs/cli.md:115; src/engines/mlx/load.zig:968-971; /tmp/ds41sweep/lanes.txt 2026-11-23T20:12:23Z -->

### The 1M-token context fits

The plugin bills memory in four parts. The engine admits a load or a request only when the sum fits.

| Part | Where the code computes it | Size for this model |
|---|---|---|
| Resident load | `residentBytes` (`plugin.zig`) calls `residentDiskBytes` (`deepseek_v41.zig`) | The pack's shards less the tensors it drops, plus the Engram tables it holds, the Engram LRU (256 MiB), the Engram partials' cache (4 GB) and the draft-only head |
| Compressed KV per token | `kvPerToken` (`bill.zig`) | 3,200 bytes a token, so 3.36 GB at 1,048,576 tokens |
| Window rings | the rings part of `prefillBytes`'s `state` term (`deepseek_v41.zig`) | 40 layers x (128 + 16) rows x 512 x 2 bytes, about 5.9 MB |
| Prompt pass | `prefillBytes` (`deepseek_v41.zig`), plus the `PF_KEEP_WIDE` reserve | Grows with the pass width. `PF_PASS_SIZE` picks the widest pass that fits the memory share; a set `--prefill-chunk` narrows it |

The KV term is small because only four layers (2, 8, 14 and 20) are KV sources. Each stores a 512-wide latent and a
128-wide index key per token in bf16, compressed 2:1 in three layers and 1:1 in layer 20.

On build `c59b60c6` (spec off), a 1,038,931-token request at a 1,048,576-token context was admitted: the admission
check needed 38,246 MB and had 78,476 MB available, with a 20.03 GB reserve armed. That build had no Engram partials'
cache. The cache bills 4 GB and allocates 3,787 MiB on this pack, so by arithmetic it leaves about 36 GB of the
40,230 MB margin. On control `04b8f7d0`, a load at a 1,064,960-token context succeeded, with a 17,769 MB reserve at
44,544-row passes, and admitted a 4-token request. A real 1M-token prompt has not run on the release head yet (see
"Gaps").

<!-- src: src/plugins/deepseek_v41/plugin.zig:937-946 (residentBytes); deepseek_v41.zig:2873-2899
     (residentDiskBytes), :2821-2845 (prefillBytes); bill.zig kvPerToken; decode/engram_kvc.zig:33 (cap 4 GB);
     env.zig:346 (LRU 256); config values from the NVFP4 pack's config.json on m1
     (~/nvpf/packs/DeepSeek-V4.1-Flash-1017-NVFP4-mlx-nvfp4: kv_source_layer_ids [2,8,14,20], compress_ratios 2 at
     2-19 and 1 at 20-39, head_dim 512, index_head_dim 128, sliding_window 128, 40 layers).
     3,200 = 2 x (256+256+256+512 + 64+64+64+128).
     m2:~/lev/pfkeep/mem1.out (c59b60c6 1M admission: needed 38246, available 78476, reserve 20.03 GB, 1038931 tokens);
     m1:~/glmx/abba/ds41-admit-1064960/run/serve.log:78 (KV cache 3787 MB), :171 (reserve 17769 MB at 44544 rows),
     :202 (admission of the 4-token request) -->

## The default configuration

The defaults are the release configuration. Each speed change in this release won its A/B on an M5 Ultra with
512 GB, except three washes kept because they remove work (the PR body has the A/Bs). The standard test is three
coding prompts of about 17,600 tokens and 1,024 greedy completion tokens. Every switch is an
environment variable `MLX_SERVE_DSV41_<NAME>`, registered with its default and one line of doc in `src/env.zig`.
A default-on switch turns off with `=0`. You do not need to set any switch to get the release configuration.

Other Macs get the same defaults. Some parts scale or turn off by themselves:

- A Mac that cannot hold the pack in memory refuses the load. With 256 GB or less, the routed experts must stream
  (per-model `stream_experts`), and decode is then limited by disk reads.
- The prompt pass width scales with memory (`PF_PASS_SIZE`).
- Where the Engram tables do not fit in memory, the plugin reads them from disk per lookup. The switches that need
  every table held then decline, and the kernel family falls back to the `ssd` path (`WEIGHT_PATH`).
- On GPUs without neural accelerators (M4 and older, or an M5 on macOS before 26.2), the matrix-unit kernels stay off.

We did not tune or measure any machine other than the M5 Ultra with 512 GB.

<!-- src: /tmp/ds41sweep/release/audit.md section 3; src/env.zig:347 (ENGRAM_RESIDENT), :364 (PF_PASS_SIZE),
     :398 (WEIGHT_PATH); src/plugins/deepseek_v41/AGENTS.md "Known issues" (non-NAX); src/env.zig:1-9 -->

## Speculative decoding (DSpark)

The pack ships DeepSeek's DSpark stages: a small drafter that proposes a block of tokens, which the model then
checks in one wide verify step. The plugin loads the stages whenever the pack ships them. `--no-mtp` turns them off,
and `MLX_SERVE_DSV4_DSPARK=0` turns drafting off. The model's verify accepts or replaces every token that the drafter
proposes. A policy picks how many rows each verify checks (`MLX_SERVE_DSPARK_POLICY2`).

<!-- src: deepseek_v41.zig:2851-2862 (dsparkWanted); src/env.zig:312, :336-337; docs/cli.md:77 -->

**Greedy requests.** The drafter proposes its argmax tokens, and the verify keeps the longest prefix that the model
agrees with. The verify rows run multi-row kernels: the same math as a plain decode step, in another sum order. So
greedy text with DSpark on can differ from plain decode by rounding. `MLX_SERVE_DSV41_EXACT_VERIFY=1` makes each
verify row run its plain step's kernels, and greedy DSpark text is then plain decode's, at some speed cost.

<!-- src: src/env.zig:348 (EXACT_VERIFY); src/plugins/deepseek_v41/AGENTS.md "Known issues" (EXACT_VERIFY on NAX) -->

**Sampled requests (`DRAFT_SAMPLE`, on).** The drafter draws each draft token from its own distribution q. The verify
keeps it with probability min(1, p / q), where p is the model's distribution, and resamples a rejected token from
max(0, p - q). This is standard speculative sampling, so the output keeps the model's distribution.

<!-- src: src/env.zig:354 (DRAFT_SAMPLE doc) -->

**The draft vocabulary (`DRAFT_VOCAB`).** With a `draft-vocab.txt` file in the pack directory, greedy drafts score only
the listed token ids. The file holds decimal ids separated by whitespace. Our runs used 65,536 ids. The trunk and
every verify keep the whole vocabulary, so only the drafts change. `DRAFT_VOCAB_PROMPT` (on) also adds up to 16,384
of the request's own most frequent prompt ids. A sampled request always drafts over the whole vocabulary, so its
drafts and tokens are the same as without the file. `MLX_SERVE_DSV41_DRAFT_VOCAB=<path>` names another file, and an
empty value or `0` turns the vocabulary off. Neither the converter nor any script in this repo writes
`draft-vocab.txt`. The file gave +1.46% decode on the standard test.

<!-- src: deepseek_v41.zig:1274-1290 (draftPromptVocab: sampled drafts every id), :1365-1372, :1421-1431
     (draftVocabPath); src/env.zig:374-375; /tmp/ds41sweep/pr/1256-body.md (65,536 ids; converter does not write it);
     audit.md 1b (+1.46%, sp-draftvocab4); git grep draft-vocab (only env.zig and deepseek_v41.zig) -->

## Lossless classes

"Lossless" here means no change of precision, quantization, routing or approximation against the pack as shipped.
A reordered sum or a changed tie-break is allowed when the result is no farther from float64. Each default-on switch
is in one class. The HEAD8 head above is the one stated exception.

| Class | What it means | Default-on switches |
|---|---|---|
| EXACT | Bit-identical to the `=0` path | 30 |
| REORDER | The same math in another sum order or tie-break. Over a test set, the total l2 error and the worst element against float64 are no farther than the `=0` path's. `MOE_PAIR_SCALED` and `PF_HC_V2` are held to the reorder bounds below instead | 14 |
| DRAFT-ONLY | Changes only the drafts. The verify accepts or replaces every draft | 5, plus `DRAFT_VOCAB` |
| TF32 | MLX's TF32 matmul switch (`MLX_ENABLE_TF32`, on). Exact on the verified path; only the drafter's P·V rounds | 1 (engine) |

<!-- src: /tmp/ds41sweep/release/audit.md (definition, class key, 1a); git show ae57f5b4 (REORDER ruling text);
     counts: env.zig:339-397 has 51 default-on switches (FLASH_ATTN a text switch) = 30 EXACT + 14 REORDER + 5 DRAFT-ONLY + HEAD8 + ATTN_F16 -->

The tests that hold these classes are in the plugin. Each test runs a planted fault as a control, and the control
must fail.

- `prompt/wide_test.zig` holds the prompt switches: word-for-word tests for the EXACT ones, and float64 tests for
  `FLASH_ATTN`, `PF_COMP_BF16`, `MOE_PAIR_SCALED` and `PF_HC_V2`.
- `decode/attn.zig`, `decode/woa_plain.zig` and `decode/engram_ids.zig` hold `ATTN_DEC`, `WOA_PLAIN` and
  `ENGRAM_GPU_IDS` against float64.
- `tf32_test.zig` runs the served f32 matmuls against float64 over operands saved from a served run
  (`DSV41_TF32_FIXTURE`).

<!-- src: git show abf5a292, 11cff435, ae57f5b4, d62ac3e4, 3679e6c6 (messages and stats);
     prompt/wide_test.zig:333, :594, :1317, :1928, :2135; tf32_test.zig:1-5, :116 -->

**Results of the float64 and word tests.**

- `MOE_TAIL` (EXACT): the one launch gives the two launches' words, with 0 differing words at 17 to 8,192 rows.
- `FLASH_ATTN` (REORDER): its f32 output is nearer to float64 than the gather's, 6.4e-5 against 1.8e-4 relative.
- `MOE_PAIR_SCALED` (REORDER, bounded): about 0.2% of the activation words differ from the split path. Each one is
  within one bf16 step of its row's largest value. Where the split path gives exactly zero, the paired kernel keeps a
  tiny value, about 1e-35. The whole-sample l2 error against float64 measured equal to the split path's. The test
  holds it to under 1% of words differing, an l2 within 1.05 times the split path's over the sample, and within 4 times
  over the differing words. The release head's full suite failed that last bound once (see "Gaps").
- `PF_HC_V2` (REORDER, bounded): an f32 reorder of the five-pass hyper-connections. Its f32 mixes and split are 2 to 6
  times farther from float64 than the five passes', at the 1e-7 level. That is below the bf16 rounding that follows,
  and the tail's output words are the five passes' words. The test holds the mixes and split within 1e-5 of float64
  and every other output within 1.01 times the five passes' l2.
- `ATTN_DEC` and `WOA_PLAIN` (REORDER): nearer to float64 than their `=0` paths in total l2 and worst element.
- `ENGRAM_GPU_IDS` (REORDER): 30 of 409,600 bf16 words differ from MLX's chain (25 nearer, 4 farther). The worst
  element is equal, and the total squared error is larger by a factor of 1 + 4.5e-9.
- TF32: the router, the compressor projection and the DSpark q·k read 0 operand elements that TF32 rounds. The hc
  collapse has operands that TF32 would round, but MLX serves it on its gemv kernel (one row a product), which does
  not use TF32, so its error is the same with TF32 on and off. The only TF32 loss is the drafter's P·V, at a relative
  l2 of 2.8e-4 to 2.9e-4, and it changes only the drafts.

<!-- src: git show abf5a292 (MOE_TAIL 0 words; FLASH_ATTN 6.4e-5 vs 1.8e-4; PF_HC_V2 3-6x, z l2 8.8e-7 vs 1.5e-7);
     git show d62ac3e4 (MOE_PAIR_SCALED 0.196-0.214%, <= 1.000 bf16 step, l2 1.0000x; PF_HC_V2 bounds, tail x2 same
     words); git show ae57f5b4 (ATTN_DEC, WOA_PLAIN, ENGRAM_GPU_IDS numbers). Checked against the release head's
     m2:~/lev/pr1256/bdw-10.log (moe-tail-words, fa-gather, hcv2-f64 2.0-5.9x, attn-dec f64 72 cells, woa-plain f64
     24 cells, engram-gpu-ids f64 :885) and m2:~/lev/rhbwide/t6.log, t8.log, t11.log (MOE_PAIR_SCALED largest word
     step 1343-1347 from a zero word: bf16 word 1347 = 9.2e-36); m2:~/lev/pr1256/full10.log:3083 (the 4x bound failed
     at 17 rows); TF32: m2:~/lev/tf32/m-on.log and m-off.log lines 26-40 (router, lin_f32, dspark_qk 0 rounded;
     hc_collapse gemv, same error on and off; dspark_pv 2.84e-4 to 2.91e-4) -->

`FP8_FOLD` and `FP4_READS` read three rare scale codes differently from MLX. At load, the plugin scans each layer for
those codes and declines both switches by name where it finds one. The shipped NVFP4 pack holds none of them.

<!-- src: git show 6c48635e (0 of 231,292,928 e8m0 scales >= 247; 0 of 7,401,373,696 e4m3 NaN) -->

<details>
<summary>Every default-on switch and its class</summary>

Names omit `MLX_SERVE_DSV41_`. "Test" names the evidence kind: a standing test in the plugin, or the text of the A/B
and the commit that added the switch.

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
| PF_ROUTER_BF16 | Router logits on a bf16 MMA, only where the single sum ran | EXACT | test |
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
| FLASH_ATTN | Wide attention spans on one flash kernel | REORDER | test |
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
| ATTN_F16 | f16 attention core | Not classed; off on M5, where a call that flash and the gather do not take runs the f32 chain (see "Gaps") | float64 measurement, printed |

Engine variables on the DeepSeek path: `MLX_SERVE_DSV4_DSPARK` (REORDER: the verify's multi-row kernels),
`MLX_SERVE_DSPARK_POLICY2` and `MLX_SERVE_DSPARK_POLICY2_MODELED` (DRAFT-ONLY: they pick verify widths) and
`MLX_ENABLE_TF32` (TF32).

<!-- src: /tmp/ds41sweep/release/audit.md 1a and E (classes, evidence), updated by commits abf5a292 (MOE_TAIL test,
     FLASH_ATTN test), 11cff435/ae57f5b4 (ATTN_DEC, WOA_PLAIN, ENGRAM_GPU_IDS tests), d62ac3e4 (bounds), 6c48635e
     (load scan); src/env.zig:339-398 -->

</details>

## Switches that stay opt-in

Each one is off by default. Names omit `MLX_SERVE_DSV41_` except where shown.

- `EXACT_VERIFY=1` (EXACT): greedy DSpark text equals plain decode's bytes on M5.
- `EXACT=1|solo` (EXACT; any value other than unset, empty or `0` turns it on): the same on M3-generation or later
  Apple GPUs without neural accelerators; on a GPU with them the load leaves it off and logs why.
- `FLASH_PSUM=1` (REORDER): a flash-attention variant that is nearer to float64; it measured no faster.
- `MXFP4_VIEW=1` (REORDER): loads an NVFP4 bank whose every 32-value span is one power of two as MXFP4.
- `MOE_PAIR=1` (REORDER): the paired prompt gate/up for MXFP4 banks without per-expert scales.
- `MLX_SERVE_DSPARK_PIN_ROWS=N` (test only): every DSpark round, greedy or sampled, verifies N rows, for width
  sweeps.

Three value switches set memory and kernel choices: `ENGRAM_LRU_MB` (256) tunes memory only. `ENGRAM_RESIDENT`
(`auto`) and `WEIGHT_PATH` (`auto`) can drop the resident fusions: the `ssd` path turns off `FUSE_MXFP8_QMM` (a
REORDER switch), and tables not held turn off `ENGRAM_GPU_IDS` and the Engram KV cache. `DRAFT_VOCAB` (above) is a
value switch too. The dead-code pass removed nine switches that lost their A/Bs or never ran one
(`rel-deadcode` `41be1c30`, merged in `b7b267fd`).

<!-- src: src/env.zig:338, :346-348, :350, :372, :376, :379, :398; deepseek_v41.zig:2601-2613 (EXACT left off on NAX);
     audit.md section 2 (FLASH_PSUM -0.07%, KEEP-OPTIN list); git log 049b7d2a..230412b7 (9 removals), b7b267fd -->

## Results

The PR body for PR 1256 holds the A/B table of every switch and its raw runs. On control `04b8f7d0`, the standard
test gave these decode speeds:

| Cell | Decode tok/s |
|---|---|
| Greedy, DSpark on | 97.43 |
| The model card's sampling settings, DSpark on | 88.06 |
| Greedy, DSpark off | 61.48 |
| Prompt p10, greedy, DSpark on | 106.45 |

Prefill was 5,548.7 tok/s in the greedy DSpark-on cell and 5,530 to 5,535 tok/s in the other three. The final
numbers on the release head are not done yet.

<!-- src: m1:~/glmx/abba/ds41-final-std-greedy/summary.txt:19, ds41-final-p10-greedy/summary.txt:15;
     m2:~/glmx/abba/ds41-final-std-card/summary.txt:19, ds41-final-std-specoff/summary.txt:19 -->

## Gaps

- HumanEval, the 1K to 1M context sweep and the final standard test on the release head have not run.
- The release must ship `draft-vocab.txt` with the pack. No tool in this repo makes it.
- We tuned and measured only on M5 Ultra Macs with 512 GB. Smaller Macs and older GPUs get the same defaults, untested.
- On Macs without neural accelerators, `ATTN_F16` (on) serves an f16 attention core that has no float64 test. A
  release for those Macs needs that test, or `ATTN_F16` off.
- `HEAD8` is a quantization change. Its KLD equals the bf16 head's on our set, but it is not bit-identical.
- At the release head, the full test suite failed the `MOE_PAIR_SCALED` bounds test once: at 17 rows, over the words
  that differ, its float64 l2 was 7.5 times the split path's, against a bound of 4. The release needs a ruling on
  that test before it ships.

<!-- src: deepseek_v41.zig attnCore and attnF16At (on NAX the f16 core only when ATTN_F16 is named on);
     audit.md table C (ATTN_F16, HEAD8) -->

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
| Neural accelerators (NAX) | The M5 GPU's matrix units |
| KLD | Kullback-Leibler divergence of the served distribution from a reference |
| Standard test | Three coding prompts of about 17,600 tokens, 1,024 greedy tokens, a fresh server per run |
